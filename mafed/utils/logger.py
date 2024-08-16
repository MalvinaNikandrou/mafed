import logging
import os
from typing import Dict, Optional

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

try:
    from wandb.wandb_run import Run

    import wandb
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run = None, None


_LOG_FMT = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
_DATE_FMT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger("__main__")  # this is the global logger


def add_log_to_file(log_path):
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


class CLWandbLogger(WandbLogger):
    @property
    def experiment(self):
        r"""

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

        .. code-block:: python

            self.logger.experiment.some_wandb_function()

        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)
                    self._experiment.define_metric(
                        "average_accuracy",
                        step_metric="trainer/valid_step",
                        step_sync=True,
                    )
                    self._experiment.define_metric("BWT", step_metric="trainer/valid_step", step_sync=True)

        return self._experiment

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        is_valid_step: bool = False,
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            if is_valid_step:
                self.experiment.log({**metrics, "trainer/valid_step": step})
            else:
                logging_step = step + self._step_offset
                self.experiment.log({**metrics, "trainer/global_step": logging_step})
        else:
            self.experiment.log(metrics)

    @rank_zero_only
    def set_global_step_offset(self, offset: int = 0) -> None:
        self._step_offset = offset
