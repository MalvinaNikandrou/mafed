from typing import Literal

import torch
from tqdm import tqdm

from mafed.utils.logger import LOGGER


class DistillationWeights:
    def __init__(
        self,
        distillation_modality_weighing_strategy: Literal[
            "equal",
            "balanced",
            "adaptive",
        ] = "equal",
        distillation_layer_weighing_strategy: Literal["single", "equal", "discounted", "cumulative"] = "single",
        gamma: float = 0.9,
        num_hidden_layers: int = 11,
        distillation_layer: int = -1,
        num_vision_tokens: int = 256,
    ) -> None:
        self._layer_coeff_method = "equal"
        # Discount factor to compute weight per layer
        # The closer gamma is to 1, the more equally the losses per layer are weighted
        self.gamma = gamma
        self.num_vision_tokens = num_vision_tokens
        self._hidden_state_layer = distillation_layer
        self._modality_weighing_strategy = distillation_modality_weighing_strategy
        if self._modality_weighing_strategy == "balanced":
            self.lang_coeff = 0.5

        if distillation_layer is None and distillation_layer_weighing_strategy == "single":
            raise AssertionError("Invalid layer weighting strategy 'single'. Use 'equal' or 'discounted' instead!")
        if distillation_layer is None and distillation_layer_weighing_strategy == "cumulative":
            raise AssertionError("Invalid layer weighting strategy 'cumulative'. Please pass the distillation layer!")
        if distillation_layer_weighing_strategy == "cumulative":
            self.num_hidden_layers = distillation_layer
        else:
            self.num_hidden_layers = num_hidden_layers
        if distillation_layer is not None and distillation_layer_weighing_strategy != "cumulative":
            distillation_layer_weighing_strategy = "single"
        self._layer_weighing_strategy = distillation_layer_weighing_strategy
        self.prepare_layer_coeffs()
        LOGGER.info(
            f"Distillation layer weighting strategy: {self._layer_weighing_strategy} layer(s): {self.get_distillation_layers()}"
        )

    def prepare_layer_coeffs(self):
        """Prepare the coefficient for the distillation loss per layer."""
        if self._layer_weighing_strategy == "single":
            self.layer_coeffs = None
            return
        if self._layer_weighing_strategy == "equal":
            self.layer_coeffs = torch.ones(self.num_hidden_layers) / self.num_hidden_layers
            return

        layer_coeffs = torch.tensor([self.gamma**distance for distance in torch.arange(self.num_hidden_layers, 0, -1)])
        layer_coeffs = layer_coeffs / layer_coeffs.sum()
        self.layer_coeffs = layer_coeffs

    def update_weights(self, model, dataloader, task_id):
        if self._modality_weighing_strategy != "adaptive":
            return
        wimportances = self.compute_adaptive_weights(model, dataloader)
        if task_id < 1:
            self.lang_coeff = wimportances
        else:
            self.lang_coeff = (wimportances + task_id * self.lang_coeff) / (task_id + 1)

    def get_modality_loss_weights(self, batch, layer: int):
        if self._modality_weighing_strategy == "equal":
            return self._get_equal_loss_weights(batch)
        elif self._modality_weighing_strategy == "balanced":
            return self._get_balanced_loss_weights()
        elif self._modality_weighing_strategy == "importance":
            return self._get_importance_layer_loss_weights(layer)
        else:
            raise NotImplementedError

    def get_distillation_layers(self):
        if self._layer_weighing_strategy == "single":
            return [self._hidden_state_layer]
        return [layer for layer in range(self.num_hidden_layers)]

    def get_layer_loss_weight(self, layer: int):
        if self.layer_coeffs is None or self._layer_weighing_strategy == "single":
            return 1.0
        return self.layer_coeffs[layer]

    def compute_adaptive_weights(self, model, dataloader):
        """
        Compute gradient-based importance matrix for each modality.
        """
        model.eval()
        hidden_layers = self.get_distillation_layers()
        total_lang_tokens = 0.0
        total_image_tokens = 0.0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing adaptive weights"):
            model.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    **batch, compute_loss=True, output_hidden_states=True, allow_input_gradients=True, return_dict=True
                )
                if batch_idx == 0:
                    lang_importances = torch.zeros(len(hidden_layers), device=outputs.loss.device)
                    image_importances = torch.zeros(len(hidden_layers), device=outputs.loss.device)
                # make sure the cls is masked
                bsz, txt_len = batch["attention_mask"].shape
                language_mask = torch.zeros(
                    (bsz, txt_len + self.num_vision_tokens),
                    dtype=batch["attention_mask"].dtype,
                ).to(batch["attention_mask"].device)
                language_mask[:, self.num_vision_tokens :] = batch["attention_mask"]
                batch["lang_masks"] = language_mask
                image_masks = torch.zeros(
                    (bsz, txt_len + self.num_vision_tokens),
                    dtype=batch["attention_mask"].dtype,
                ).to(batch["attention_mask"].device)
                image_masks[:, : self.num_vision_tokens] = 1
                batch["image_masks"] = image_masks
                for index, layer in enumerate(hidden_layers):
                    grad = torch.autograd.grad(
                        outputs=outputs.loss,
                        inputs=outputs.hidden_states[layer],
                        grad_outputs=None,
                        retain_graph=True,
                        create_graph=False,
                        only_inputs=True,
                    )[0]
                    grad_norm = torch.linalg.norm(grad, dim=-1)
                    if batch_idx == 0:
                        lang_importances[index] = (grad_norm * batch["lang_masks"]).sum()
                        image_importances[index] = (grad_norm * batch["image_masks"]).sum()
                    else:
                        lang_importances[index] += (grad_norm * batch["lang_masks"]).sum()
                        image_importances[index] += (grad_norm * batch["image_masks"]).sum()

                total_lang_tokens += batch["lang_masks"].sum()
                total_image_tokens += batch["image_masks"].sum()

        lang_importances /= total_lang_tokens
        image_importances /= total_image_tokens
        lang_importances /= lang_importances + image_importances
        model.zero_grad()
        return lang_importances

    def _get_equal_loss_weights(self, batch):
        num_text_tokens = batch["lang_masks"].sum()
        num_vision_tokens = batch["image_masks"].sum()
        num_total_tokens = num_text_tokens + num_vision_tokens

        lang_weight = num_text_tokens / num_total_tokens
        vision_weight = num_vision_tokens / num_total_tokens
        return lang_weight, vision_weight

    def _get_dynamic_loss_weights(self, loss_weights):
        if loss_weights is None:
            raise ValueError("Did not get loss weights from model")

        lang_weight = loss_weights[0]
        vision_weight = 1 - loss_weights[0]
        return lang_weight, vision_weight

    def _get_balanced_loss_weights(self):
        return self.lang_coeff, (1 - self.lang_coeff)

    def _get_adaptive_layer_loss_weights(self, layer):
        if self.lang_coeff.shape[0] == 1:
            lang_weight = self.lang_coeff.item()
        else:
            lang_weight = self.lang_coeff[layer].item()
        vision_weight = 1 - lang_weight
        return lang_weight, vision_weight
