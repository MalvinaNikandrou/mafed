import glob
import json
import os
from typing import Any, Optional

import timm
import torch
from safetensors.torch import load_model
from torch.nn import CrossEntropyLoss
from torch.utils import checkpoint
from transformers import (
    AutoConfig,
    CLIPVisionModel,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, is_flash_attn_2_available
from transformers.utils.hub import cached_file

from mafed.utils.logger import LOGGER


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """Returns a moderately tiny value for a given PyTorch data type.

    This is used to avoid numerical issues such as division by zero. This is different from
    `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs. Only supports floating point
    dtypes. Implementation from AllenNLP:
    https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L2010-L2024
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in {torch.float, torch.double}:
        return 1e-13  # noqa: WPS432
    elif dtype == torch.half:
        return 1e-4  # noqa: WPS432
    raise TypeError(f"Does not support dtype {str(dtype)}")


def masked_mean(vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """To calculate mean along certain dimensions on masked values.

    Implementation from AllenNLP:
    https://github.com/allenai/allennlp/blob/39c40fe38cd2fd36b3465b0b3c031f54ec824160/allennlp/nn/util.py#L351-L377
    Args:
        vector (torch.Tensor): The vector to calculate mean.
        mask (torch.Tensor): The mask of the vector. It must be broadcastable with vector.
        dim (int): The dimension to calculate mean
        keepdim (bool): Whether to keep dimension
    Returns:
        (torch.Tensor): Masked mean tensor
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)  # noqa: WPS358

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def average_task_loss(labels: torch.Tensor, logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Average the loss with respect to the sequence length for each element in the batch.

    This is used so that loss for smaller sequences is not penalized:
    1) Compute the cross-entropy loss a
    2) Average by sequence length
    3) Average by batch size
    """
    (bsz, seq_len) = labels.size()
    loss_fct = CrossEntropyLoss(reduction="none")

    labels_mask = labels != -100
    # flat_labels shape (batch_size, seq_len) -> (batch_size * seq_len)
    flat_labels = labels.view(-1)
    # flat_logits shape (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    flat_logits = logits.view(-1, vocab_size)
    # loss shape (batch_size, seq_len)
    loss = loss_fct(flat_logits, flat_labels).view(bsz, seq_len)
    # averages over the sequence length dimension first and then over the batch dimension
    return masked_mean(loss, labels_mask, dim=-1).mean()


def compute_loss(labels: torch.tensor, logits: torch.tensor, vocab_size: int) -> torch.tensor:
    """Compute loss averaged across the sequence length."""
    labels = labels.to(logits.device)  # type: ignore[assignment]
    logits = logits[:, -labels.size(1) :, :]
    # Shift so that tokens < n predict n and enable model parallelism
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(logits.device)

    # Compute the loss averaged over the sequence length for each element in the batch
    loss = average_task_loss(shift_labels, shift_logits, vocab_size)
    return loss


def load_config_hf(model_name: str) -> Any:
    """Load model configuration."""
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    with open(resolved_archive_file, "r") as fp:
        data = json.load(fp)
    return data


def load_state_dict_hf(
    model_name: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """Load model state dict."""
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in {torch.float32, None} else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)


class PretrainedEva02(timm.models.Eva):
    """Pretrained EVA_02 model."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()

    @classmethod
    def from_pretrained_model(cls, model) -> "PretrainedEva02":
        """Create model from pretrained model."""
        new_model = cls()
        new_model.__dict__ = model.__dict__.copy()
        return new_model

    def forward_features(self, x):
        """Forward features."""
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        outputs = self.forward_features(inputs)
        return outputs

    def get_num_layer(self, var_name: str) -> int:
        """Get number of layers."""
        if var_name in {"cls_token", "mask_token", "pos_embed"}:
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split(".")[1])
            return layer_id + 1
        return len(self.blocks)


def create_eva2_model(eva_name: str, img_size: int = 336) -> PretrainedEva02:
    """Create EVA_02 model."""
    pretrained_model = timm.create_model(
        eva_name,
        pretrained=True,
        img_size=img_size,
    )
    model = PretrainedEva02.from_pretrained_model(pretrained_model)

    # use_fused_attention = timm.layers.use_fused_attn()
    # LOGGER.info(f"Using fused attention: {use_fused_attention}")
    return model


def build_vision_encoder(
    vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
) -> PreTrainedModel:
    """Build vision encoder."""
    if "eva" in vision_encoder_name:
        LOGGER.info(f"Loading EVA_02 pretrained: {vision_encoder_name}")
        vision_encoder = timm.create_model(
            vision_encoder_name,
            pretrained=True,
            num_classes=0,
        )
        # Remove the head since we only want the features
        # This will save some memory from the gpu
        # del vision_encoder.head
        # This is a workaround to prevent ovewriting the weights of the vision
        # encoder in post_init() call
        for module_name, module in vision_encoder.named_modules():
            module._is_hf_initialized = True

    elif "clip" in vision_encoder_name:
        LOGGER.info(f"Loading CLIP pretrained: {vision_encoder_name}")
        vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_name)
    else:
        raise ValueError(f"Unknown vision encoder: {vision_encoder_name}")
    return vision_encoder


class VLCLIPGPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    """VLGPTNeoXForCausalLM model."""

    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vision_encoder = build_vision_encoder(config.vision_encoder_name)

        self._select_layer = config.select_layer
        self._select_feature = config.select_feature

        vision_encoder_hidden_size = (
            self.vision_encoder.config.hidden_size
            if isinstance(self.vision_encoder, CLIPVisionModel)
            else self.vision_encoder.num_features
        )

        modules = [
            torch.nn.Linear(
                vision_encoder_hidden_size,
                config.hidden_size,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        ]
        self.vision_embed_tokens = torch.nn.Sequential(*modules)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> torch.nn.Linear:  # noqa: WPS615
        """Get output embeddings."""
        return self.embed_out

    def set_output_embeddings(self, new_embeddings: torch.nn.Linear) -> None:  # noqa: WPS615
        """Set output embeddings."""
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.tensor] = None,
        pixel_values: Optional[torch.tensor] = None,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        inputs_embeds: Optional[torch.tensor] = None,
        head_mask: Optional[torch.tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.tensor]]] = None,
        labels: Optional[torch.tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        allow_input_gradients: bool = False,
        **kwargs,
    ):
        """Foward pass."""
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        patch_embeddings = self.get_patch_embeddings(pixel_values)

        # step 2: project the visual features to the size of textual embeddings
        language_model_inputs = self.vision_embed_tokens(patch_embeddings)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )

        if input_ids is None:
            inputs_embeds = language_model_inputs
            attention_mask = language_model_attention_mask
        else:
            # step 3: concatenate the visual embeddings and the textual embeddings
            inputs_embeds = self.gpt_neox.embed_in(input_ids)
            inputs_embeds = torch.cat(
                [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
                dim=1,
            )

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            expected_device = language_model_attention_mask.device
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if allow_input_gradients:
            inputs_embeds = inputs_embeds.detach()
            inputs_embeds.requires_grad = True
        # step 4: forward through the language model
        outputs = self.gpt_neox(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            lm_loss = compute_loss(labels, lm_logits, self.config.vocab_size)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.tensor,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.tensor] = None,
        inputs_embeds: Optional[torch.tensor] = None,
        pixel_values: Optional[torch.tensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        input_shape = input_ids.shape
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # print(input_ids.shape, pixel_values.shape, attention_mask.shape, inputs_embeds)
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": None,
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "use_cache": kwargs.get("use_cache", False),
            }
        )

        return model_inputs

    @classmethod
    def from_pretrained(  # noqa: WPS231
        cls,
        pretrained_model_name: str,
        vision_encoder_name: str = "openai/clip-vit-base-patch32",
        select_layer: int = -2,
        select_feature: str = "patch",
        use_flash_attention_2: bool = False,  # noqa: WPS114
        state_dict=None,
    ):
        """Load model from pretrained model name."""
        use_flash_attention_2 = use_flash_attention_2 and is_flash_attn_2_available()  # noqa: WPS114
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

        if os.path.exists(pretrained_model_name) and os.path.isdir(pretrained_model_name):
            config_data = load_config_hf(pretrained_model_name)
            config = PretrainedConfig(**config_data)
            config.vision_encoder_name = vision_encoder_name
            config.select_layer = select_layer
            config.select_feature = select_feature
            config._attn_implementation = attn_implementation  # noqa: WPS437
            model = cls(config=config)
            LOGGER.info(f"Loading model from {pretrained_model_name}")
            if os.path.exists(os.path.join(pretrained_model_name, "model.safetensors")):
                LOGGER.info("Attempting to load model using safetensors")
                try:
                    load_model(model, os.path.join(pretrained_model_name, "model.safetensors"))
                    if state_dict is not None:
                        model.load_state_dict(state_dict=state_dict, strict=False)
                    return model
                except Exception as safetensor_exception:
                    LOGGER.error(f"Failed to load model using safetensors: {safetensor_exception}")

            if len(glob.glob(os.path.join(pretrained_model_name, "*.safetensors"))) >= 2:
                LOGGER.info("Found multiple safetensors files, using sharded checkpoint loading")
                try:
                    load_sharded_checkpoint(model, pretrained_model_name)
                    if state_dict is not None:
                        model.load_state_dict(state_dict=state_dict, strict=False)
                    return model
                except Exception as safetensor_exception:
                    LOGGER.error(f"Failed to load model using safetensors: {safetensor_exception}")

            if os.path.exists(os.path.join(pretrained_model_name, "pytorch_model.bin")):
                LOGGER.info("Attempting to load model using torch.load")
                try:
                    model.load_state_dict(torch.load(os.path.join(pretrained_model_name, "pytorch_model.bin")))
                    if state_dict is not None:
                        model.load_state_dict(state_dict=state_dict, strict=False)
                    return model
                except Exception as torch_exception:
                    LOGGER.error(f"Failed to load model using torch.load: {torch_exception}")
            else:
                LOGGER.error(f"Could not load model from {pretrained_model_name}")

        else:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.vision_encoder_name = vision_encoder_name
            config.select_layer = select_layer
            config.select_feature = select_feature
            config._attn_implementation = attn_implementation  # noqa: WPS437
            model = cls(config=config)
            model.load_state_dict(load_state_dict_hf(pretrained_model_name), strict=False)

            if state_dict is not None:
                model.load_state_dict(state_dict=state_dict, strict=False)
        return model

    def get_patch_embeddings(self, pixel_values: torch.tensor) -> torch.tensor:
        """Get patch embeddings."""
        if isinstance(self.vision_encoder, CLIPVisionModel):
            patch_embeddings = self.vision_encoder(pixel_values, output_hidden_states=True)
        else:
            patch_embeddings = self.vision_encoder.forward_features(pixel_values)

        patch_embeddings = self.feature_select(patch_embeddings)
        return patch_embeddings

    def feature_select(self, image_forward_outs: Any) -> torch.tensor:
        """Feature select."""
        # TODO: feature selection from layer is only supported for CLIP
        if isinstance(self.vision_encoder, CLIPVisionModel):
            image_features = image_forward_outs.hidden_states[self._select_layer]
        else:
            image_features = image_forward_outs

        if self._select_feature == "patch":
            return image_features[:, 1:]
        elif self._select_feature == "cls_patch":
            return image_features
        raise ValueError(f"Unexpected select feature: {self._select_feature}")

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.tensor):
        reordered_past = ()
        for layer_past in past_key_values:  # noqa: WPS519
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
