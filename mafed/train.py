"""
Continual Learning finetuning for VQA
"""

import argparse
import os
from os.path import join

import numpy as np
import timm
import torch
import transformers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from torch.cuda.amp.grad_scaler import GradScaler

from mafed.constants import PATIENCE_THRESHOLD
from mafed.data.loader import PrefetchLoader
from mafed.dataloaders import TaskDataModule, get_val_dataloaders, prepare_train_dataset
from mafed.methods import CLMethod
from mafed.model.vqa_cont_learner import VLPythiaVQACLearner
from mafed.pretrain_vlpythia import ModelArguments, build_tokenizer
from mafed.utils.checkpoint import get_initialization_checkpoint, load_model_from_checkpoint
from mafed.utils.cl_utils import random_task_order
from mafed.utils.eval_utils import validate_pythia_vqa
from mafed.utils.logger import LOGGER, CLWandbLogger
from mafed.utils.misc import parse_with_config
from mafed.utils.save import save_configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ContinualLearningTrainer:

    def __init__(self, opts) -> None:
        self.config = opts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(opts.seed)
        self.wandb_logger = self.setup_wandb_logger()
        self._initialize_tasks()
        save_configs(self.config)

    @torch.no_grad()
    def validate_all_tasks(
        self,
        model,
        task_id,
        accuracy,
        tokenizer,
    ):
        """Validate all tasks after training for a task."""
        model.eval()
        for val_task_id, val_task in enumerate(self.config.tasks):
            LOGGER.info(val_task)
            val_log, _ = validate_pythia_vqa(
                LOGGER,
                model,
                self.val_dataloaders[val_task],
                tokenizer=tokenizer,
            )
            accuracy[val_task_id, task_id] = val_log["valid/acc"]

        metrics = {"validation/average_accuracy": np.mean(accuracy[:, task_id])}
        LOGGER.info(f"Average score: {metrics['validation/average_accuracy'] * 100:.2f}")
        if task_id > 0:
            metrics["validation/BWT"] = np.mean(np.diag(accuracy[:task_id, task_id] - accuracy[:task_id, :task_id]))
            LOGGER.info(f"Average forgetting: {metrics['validation/BWT'] * 100:.2f}")
        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(metrics, step=task_id, is_valid_step=True)
        return accuracy
        return None

    def setup_wandb_logger(self):
        wandb_logger = CLWandbLogger(
            project=self.config.run_project,
            entity=self.config.run_entity,
            group=self.config.run_group,
            name=self.config.run_name,
            offline=False,
        )
        wandb_logger.set_global_step_offset(0)
        return wandb_logger

    def get_tokenizer(self):
        # load DBs and image dirs
        model_args = ModelArguments(
            model_name=self.config.model_name,
            tokenizer_name=self.config.tokenizer_name,
            vision_encoder_name=self.config.vision_encoder_name,
            tokenizer_padding_side="left",
        )

        return build_tokenizer(model_args=model_args)

    def get_image_preprocessor(self):
        data_cfg = {
            "input_size": (3, 224, 224),
            "interpolation": "bicubic",
            "mean": (0.48145466, 0.4578275, 0.40821073),
            "std": (0.26862954, 0.26130258, 0.27577711),
            "crop_pct": 0.9,
            "crop_mode": "center",
        }
        return timm.data.create_transform(**data_cfg)

    def get_prev_best_model(self, task_id, task, checkpoint_extension=".ckpt"):
        """
        task_id: The idx of the current task in [0, len(tasks)-1]
        """
        if task_id == 0 and self.config.start_task_idx > 0 and self.config.checkpoint_dir:
            prev_best_model = join(self.config.checkpoint_dir, f"{task}_best{checkpoint_extension}")
        else:
            prev_best_model = join(self.config.output_dir, "ckpt", f"{task}_best{checkpoint_extension}")
        return prev_best_model

    def initialize_continual_learning_method(self, model_config, scaler):
        # Initialize the CL method
        LOGGER.info(f"CL Method = {self.config.cl_method}")
        cl_method = CLMethod[self.config.cl_method](
            opts=self.config,
            memory_size=self.config.cl_memory,
            model_type=self.config.model_type,
            scaler=scaler,
            reg_lambda=self.config.reg_lambda,  # EWC regularization weight
            replay_coeff=self.config.replay_coeff,  # Replay loss weight
            distillation_coeff=self.config.distillation_coeff,  # Feature distillation loss weight
            distillation_modality_weighing_strategy=self.config.distillation_modality_weighing_strategy,
            distillation_layer_weighing_strategy=self.config.distillation_layer_weighing_strategy,
            distillation_layer=self.config.distillation_layer,
            cls_distillation=self.config.cls_distillation,
            distillation_loss=self.config.distillation_loss,
            gamma=self.config.distillation_layer_discount,
            num_hidden_layers=model_config.num_hidden_layers - 1,
        )
        return cl_method

    def main(self):
        """Main CL training loop."""
        tokenizer = self.get_tokenizer()
        model_config = transformers.AutoConfig.from_pretrained(self.config.model_name)
        image_preprocessor = self.get_image_preprocessor()

        self.val_dataloaders = get_val_dataloaders(
            config=self.config,
            tokenizer=tokenizer,
            image_preprocessor=image_preprocessor,
        )

        scaler = GradScaler()
        cl_method = self.initialize_continual_learning_method(model_config, scaler)
        accuracy = np.zeros((len(self.config.tasks), len(self.config.tasks)))
        prev_best_model = get_initialization_checkpoint(
            self.config, task_id=0, checkpoint_extension=self.config.init_ckpt_extension
        )

        for task_id, task in enumerate(self.config.tasks):
            LOGGER.info(f"Task {task_id}: {task}")
            # Prepare for training for the new task
            # 1. The Data
            LOGGER.info(f"Loading train data...")
            task_datamodule = TaskDataModule(
                self.config,
                task=task,
                tokenizer=tokenizer,
                image_preprocessor=image_preprocessor,
                val_dataloaders=self.val_dataloaders,
            )
            # 2. The Continual Learning Method
            cont_learner = self._prepare_continual_learner(
                task_id=task_id,
                prev_best_model=prev_best_model,
                tokenizer=tokenizer,
                cl_method=cl_method,
            )
            # 3. The Trainer
            trainer = self._prepare_trainer(task, task_id)
            # 4. The Best Model Path
            prev_best_model = self.get_prev_best_model(task_id=task_id, task=task)
            # Ready to train
            if task_id >= self.config.start_task_idx:
                cl_method.update_after_new_task(
                    model=cont_learner.model,
                    dataset=prepare_train_dataset(
                        config=self.config,
                        task=task,
                        tokenizer=tokenizer,
                        image_preprocessor=image_preprocessor,
                    ),
                )
                trainer.fit(cont_learner, task_datamodule)
            else:
                # Skip the training if the start task id is not reached
                task_datamodule.setup("train")
            self.wandb_logger.set_global_step_offset(trainer.global_step)
            del trainer
            # Load the best checkpoint
            LOGGER.info(f"Evaluating all tasks with {prev_best_model}")
            model = load_model_from_checkpoint(
                checkpoint=prev_best_model,
                model_name=self.config.model_name,
                vision_encoder_name=self.config.vision_encoder_name,
                device=self.device,
            )

            # Update the CL mehtod
            if task_id < len(self.config.tasks) - 1:
                cl_method = cont_learner.cl_method
                cl_method.update(
                    model=model,
                    dataset=task_datamodule.train_dataloader().dataset,
                    dataloader=PrefetchLoader(task_datamodule.train_dataloader()),
                    scaler=scaler,
                )
            del cont_learner
            # Run the evaluation
            accuracy = self.validate_all_tasks(
                model=model,
                task_id=task_id,
                accuracy=accuracy,
                tokenizer=tokenizer,
            )

            del task_datamodule
            del model

    def _initialize_tasks(self):
        if not self.config.tasks:
            self.config.tasks = random_task_order(self.config.exp, self.config.val_txt_dbs[0])
        if self.config.start_task_idx < 0 or self.config.start_task_idx >= len(self.config.tasks):
            raise AssertionError(f"Invalid start_task_idx: {self.config.start_task_idx}")
        LOGGER.info(f"Task order: {self.config.tasks}")
        # Get the checkpoint for the first task
        if self.config.checkpoint_dir:
            self.config.checkpoint = join(
                self.config.checkpoint_dir, f"{self.config.tasks[0]}_best{self.config.init_ckpt_extension}"
            )

    def _get_epochs(self, task_id):
        if task_id == 0:
            return self.config.epochs[0]
        return self.config.epochs[1]

    def _prepare_model_checkpoint_callback(self, task_id, task):
        model_checkpoint = ModelCheckpoint(
            monitor=f"task_{task_id}/valid_acc",
            mode="max",
            save_top_k=1,
            verbose=False,
            dirpath=join(self.config.output_dir, "ckpt"),
            filename=f"{task}_best",
            auto_insert_metric_name=False,
            save_weights_only=True,
        )
        return model_checkpoint

    def _prepare_early_stopping_callback(self, task_id):
        early_stopping = EarlyStopping(
            monitor=f"task_{task_id}/valid_acc",
            mode="max",
            patience=self.config.patience,
            min_delta=PATIENCE_THRESHOLD,
        )
        return early_stopping

    def _prepare_continual_learner(self, task_id: int, prev_best_model: str, tokenizer, cl_method):
        # Malke sure to load previous checkpoint if task_id > 0
        if task_id > 0:
            cont_learner = VLPythiaVQACLearner.load_from_checkpoint(
                prev_best_model,
                task_id=task_id,
                opts=self.config,
                tokenizer=tokenizer,
                cl_method=cl_method,
            )
        else:
            cont_learner = VLPythiaVQACLearner(
                task_id=task_id,
                opts=self.config,
                tokenizer=tokenizer,
                cl_method=cl_method,
            )
        return cont_learner

    def _prepare_trainer(self, task, task_id):
        trainer = Trainer(
            max_epochs=self._get_epochs(task_id),
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            gradient_clip_val=self.config.grad_norm,
            devices=self.config.gpus,
            accelerator="gpu",
            logger=self.wandb_logger,
            num_sanity_val_steps=0,
            enable_model_summary=True,
            callbacks=[
                RichProgressBar(),
                self._prepare_early_stopping_callback(task_id),
                self._prepare_model_checkpoint_callback(task_id, task),
            ],
            precision="bf16",
        )
        return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--config", default="config/train-vqa-base-cl-local-vlpythia.json", help="JSON config files")
    parser.add_argument(
        "--model_config",
        default="config/vlpythia-base.json",
        type=str,
        help="json file for model architecture",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    # Checkpoint parameters
    parser.add_argument("--checkpoint", default=None, type=str, help="pretrained model")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="resume training")
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        help="directory with pretrained task models",
    )
    # Preprocessing parameters
    parser.add_argument(
        "--max_txt_len",
        type=int,
        default=60,
        help="max number of tokens in text (BERT BPE)",
    )
    # Training parameters
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Total batch size for training. Not the effective batch size",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The max learning rate for Adam.",
    )
    parser.add_argument("--lr_mul", default=10.0, type=float, help="multiplier for top layer lr")
    parser.add_argument("--lr_schedule", default="triangular", type=str, help="lr scheduling")
    parser.add_argument("--epochs", default=[15, 15], type=int, nargs="+", help="Number of training epochs")
    parser.add_argument("--optim", default="adam", choices=["adam", "adamax", "adamw"], help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs="+", help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float, help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay (L2) regularization")
    parser.add_argument(
        "--grad_norm",
        default=2.0,
        type=float,
        help="gradient clipping (-1 for no clipping)",
    )
    parser.add_argument(
        "--warmup_perc",
        default=0.1,
        type=float,
        help="Percentage of training steps to linear learning rate warmup for. (linear decay)",
    )
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--n_workers", type=int, default=4, help="number of data workers")
    parser.add_argument("--pin_mem", action="store_true", help="pin memory")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--checkpoint_extension", default=".ckpt")
    parser.add_argument(
        "--start_task_idx",
        type=int,
        default=0,
        help="Which task to start training from. 0 means that we will train for the first task, "
        "idx >=1 means that we will train for the idx-th task",
    )
    parser.add_argument(
        "--exp",
        choices=[
            "diverse_domains",
            "taxonomy_domains",
            "question_types",
        ],
        default="question_types",
        help="Experiment name. This should match the [exp]_splits.json file defining the tasks",
    )
    # CL parameters
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--tasks", nargs="+", help="Task config from volta")
    parser.add_argument(
        "--cl_method",
        choices=[
            "naive",
            "ewc",
            "replay",
            "featdistill",
        ],
        default="naive",
        help="CL method",
    )
    # EWC parameters
    parser.add_argument("--reg_lambda", type=float, default=1.0, help="Weight of the EWC regularization")
    # Replay parameters
    parser.add_argument("--cl_memory", type=int, default=4000, help="Total memory size for replay")
    parser.add_argument(
        "--replay_coeff",
        type=float,
        default=1.0,
        help="Weight of the replay loss",
    )
    parser.add_argument(
        "--replay_interval",
        type=int,
        default=4,
        help="How often to replay samples from previous tasks",
    )
    # Feature distillation parameters
    parser.add_argument(
        "--distillation_modality_weighing_strategy",
        choices=["equal", "balanced", "adaptive"],
        default="equal",
        help="How to weigh the feature distillation loss per modality",
    )
    parser.add_argument(
        "--distillation_layer_weighing_strategy",
        choices=["single", "equal", "discounted", "cumulative"],
        default="single",
        help="How to weigh the feature distillation loss per layer",
    )
    parser.add_argument(
        "--distillation_coeff",
        type=float,
        default=1.0,
        help="Weight of the distillation loss",
    )
    parser.add_argument(
        "--distillation_layer_discount",
        type=float,
        default=0.9,
        help="Discount factor per layer",
    )
    parser.add_argument(
        "--distillation_layer",
        type=int,
        help="Layer to use for distillation",
    )
    parser.add_argument(
        "--distillation_loss",
        choices=["cosine", "mse"],
        default="mse",
        help="Function to use for feature distillation",
    )
    parser.add_argument(
        "--cls_distillation",
        action="store_true",
        help="Use cls token for distillation",
    )
    # Wandb parameters
    parser.add_argument("--run_entity", help="Wandb run entity")
    parser.add_argument("--run_project", default="continual-vl-pythia-finetune", help="Wandb project name")
    parser.add_argument("--run_group", help="Wandb run group")
    parser.add_argument("--run_name", help="Wandb run name")
    # Pretrained model
    parser.add_argument("--model_type", default="vlpythia")
    parser.add_argument("--model_name", default="storage/models/vl-pythia-eva-1b", help="The model checkpoint to load")
    parser.add_argument("--tokenizer_name", default="EleutherAI/pythia-410m")
    parser.add_argument("--vision_encoder_name", default="timm/eva02_large_patch14_clip_224")
    args = parse_with_config(parser)
    output_exists_not_ok = args.start_task_idx == 0 or (args.start_task_idx == 1 and args.checkpoint_dir is not None)

    if args.checkpoint and args.checkpoint_dir:
        raise ValueError("You can set either a checkpoint or a checkpoint directory, but not both.")

    ContinualLearningTrainer(args).main()
