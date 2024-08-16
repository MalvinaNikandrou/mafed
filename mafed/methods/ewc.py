import torch
from tqdm import tqdm

from mafed.methods import CLStrategy


def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    """

    return {k: torch.zeros_like(p).to(p.dtype) for k, p in model.named_parameters() if p.requires_grad}


class EWC(CLStrategy):
    """Online EWC regularizer.

    Implementation following after:
    https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/ewc.py
    At the time of development, avalanche coulnd not be directly integrated
    with the codebase due to the soft multi-label formulation of VQA.

    :param reg_lambda : (float) Weight of the EWC regularization term.
    :param online: (bool) If true keep a single sum of importances,\
            else keep importances per task.
    :param online_fator: (float)
        imp = online_factor * old_importances + new_importances
    :param soft_targets: (bool) If false, map soft targets to 1.
        Introduced for VQA.
    :param soft_targets_thres: (float) Threshold for mapping a soft target to 1.
        Used only if soft_targets is false.
    """

    def __init__(
        self,
        reg_lambda=1.0,
        online=True,
        online_factor=0.95,
        soft_targets=True,
        soft_targets_thres=0.1,
        **kwargs,
    ):
        super().__init__(reg_lambda)
        self.fisher = {}
        self.old_params = {}
        self.online = online
        self.online_factor = online_factor
        self.soft_targets = soft_targets
        self.soft_targets_thres = soft_targets_thres

    def update(self, model, dataloader, **kwargs):
        wimportances = self.compute_importances(model, dataloader)
        prev_w = {k: p.data.clone() for k, p in model.named_parameters()}
        if self.online:
            if self.task_id <= 1:
                self.fisher[0] = wimportances
            else:
                for k in self.fisher[0]:
                    self.fisher[0][k] = wimportances[k] + self.online_factor * self.fisher[0][k]

            self.old_params[0] = prev_w

        else:
            self.fisher[self.task_id] = wimportances
            self.old_params[self.task_id] = prev_w
        self.task_id += 1

    def compute_importances(self, model, dataloader):
        """
        Compute EWC importance matrix for each parameter
        """
        model.train()
        # list of list
        importances = zerolike_params_dict(model)
        total_samples = 0.0
        pbar = tqdm(dataloader, total=len(dataloader), desc="Computing importances")
        for batch in pbar:
            model.zero_grad()
            batch_size = batch["input_ids"].size(0)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = batch_size * model(**batch, compute_loss=True, return_dict=True).loss
                # Use scaler for numerical stability
                loss.backward()

            for k, p in model.named_parameters():
                try:
                    if p.grad is not None:
                        importances[k] += p.grad.data.clone().pow(2)
                    elif p.grad is None and p.requires_grad is True:
                        print("Parameter not used:", k, p.shape)
                except:
                    print("Not able to compute grad", p)
            total_samples += batch_size

        for k in importances:
            importances[k] /= total_samples

        model.zero_grad()
        return importances

    def compute_regularization(self, model, loss, task_id):
        """Compute the EWC regularization tem."""
        for k, cur_param in model.named_parameters():
            if not cur_param.requires_grad:
                continue
            old_param = self.old_params[task_id][k]
            imp = self.fisher[task_id][k]
            diff = (cur_param - old_param).pow(2)
            loss += 0.5 * self.reg_lambda * (imp * diff).sum()

        return loss

    def compute_loss(self, model, loss, **kwargs):
        if self.task_id == 0:
            return loss

        if self.online:
            loss = self.compute_regularization(model, loss, 0)
        else:
            for t in range(self.task_id):
                loss = self.compute_regularization(model, loss, t)

        return loss
