import re
from collections import Counter
from os.path import exists, join, splitext
from time import time
from typing import Any, Union

import torch
import torch.distributed as dist
from overrides import overrides
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torchmetrics import Metric
from tqdm import tqdm

from mafed.data.vqa_utils import normalize_answer


def get_checkpoint_path(task_id, task, checkpoint_dir, extension=".ckpt"):
    best_model = join(checkpoint_dir, f"{task}_best{extension}")

    if task_id == 0 and not exists(best_model):
        best_model = join(
            re.split("_ewc|_lwf|_er|_ps_|_der_|_agem|_replay|_featdistill", checkpoint_dir)[0],
            f"ckpt/{task}_best{splitext(best_model)[-1]}",
        )
    return best_model


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def compute_score(scores, labels):
    scores = torch.max(scores, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, scores.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


class VQAAccuracy(Metric):
    """Loss for VQA."""

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "total_score",
            default=torch.tensor(0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update loss sum and number of task samples."""
        if probs.shape[0] > 0:
            sample_score = compute_score(probs, targets)
            self.total_score += sample_score.sum().item()
            self.total += probs.shape[0]

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.total_score / self.total  # type: ignore[operator]


def vqa_v2_score(count: int) -> float:
    """VQA-v2 includes 10 answers for each question.

    Scores are assigned as follows:
    - 0.3 if the answer appears once
    - 0.6 if the answer appears twice
    - 0.9 if the answer appears three times
    - 1.0 if the answer appears more than three times
    """
    return min(1.0, round(0.3 * count, 1))  # noqa: WPS432


class VQAGenerativeAccuracy(Metric):
    """VQAv2 accuracy."""

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("accuracy", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, predicted_anwers: list[str], ground_truth_batch: list[list[str]]) -> None:
        """Update loss sum and number of task samples."""
        for predicted_answer, ground_truth_answers in zip(predicted_anwers, ground_truth_batch):
            predicted_answer = normalize_answer(predicted_answer)
            ground_truth_counts = Counter(ground_truth_answers)
            self.accuracy += torch.tensor(vqa_v2_score(ground_truth_counts.get(predicted_answer, 0)))

        self.total += torch.tensor(len(ground_truth_batch))

    def compute(self) -> Union[torch.Tensor, Any]:
        """Compute the total task loss."""
        return self.accuracy.float() / self.total  # type: ignore[operator]


@torch.no_grad()
def validate_vqa(logger, model, val_loader, label2ans, task_mask=None):
    model.eval()
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        targets = batch["targets"]
        with autocast():
            scores = model(batch, compute_loss=False).logits
            loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")

        probs = torch.sigmoid(scores)
        if task_mask is not None:
            loss *= task_mask
            probs *= task_mask

        val_loss += loss.sum().item()
        sample_score = compute_score(probs, targets)
        tot_score += sample_score.sum().item()

        answers = [label2ans[i] for i in probs.max(dim=-1, keepdim=False)[1].cpu().tolist()]
        for qid, answer, score in zip(batch["qids"], answers, sample_score):
            results[qid] = {"answer": answer, "acc": score.max().cpu().item()}
        n_ex += len(batch["qids"])

    if dist.is_available() and dist.is_initialized():
        metrics = torch.tensor([n_ex, val_loss, tot_score]).float().cuda()
        dist.all_reduce(metrics)
        n_ex, val_loss, tot_score = metrics

    tot_time = time() - st
    if isinstance(n_ex, torch.Tensor):
        n_ex = n_ex.item()
    if isinstance(val_loss, torch.Tensor):
        val_loss = val_loss.item()
    if isinstance(tot_score, torch.Tensor):
        tot_score = tot_score.item()
    val_loss = val_loss / n_ex
    val_acc = tot_score / n_ex
    logger.info(f"Tested {n_ex} samples")
    val_log = {
        "valid/loss": val_loss,
        "valid/acc": val_acc,
        "valid/ex_per_s": n_ex / tot_time,
        "valid/n_ex": n_ex,
    }
    model.train()
    logger.info(f"validation finished in {int(tot_time)} seconds, " f"score: {val_acc * 100:.2f}")
    return val_log, results


@torch.no_grad()
def validate_pythia_vqa(logger, model, val_loader, tokenizer):
    model.eval()
    n_ex = 0
    st = time()
    results = {}
    vqa_accuracy = VQAGenerativeAccuracy()
    for i, batch in enumerate(val_loader):
        with autocast():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                max_new_tokens=10,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        predictions = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        vqa_accuracy(predictions, batch["answers"])
        n_ex += len(batch["qids"])

    tot_time = time() - st
    if isinstance(n_ex, torch.Tensor):
        n_ex = n_ex.item()

    logger.info(f"Tested {n_ex} samples")
    val_acc = vqa_accuracy.compute()
    val_log = {
        "valid/acc": val_acc,
        "valid/ex_per_s": n_ex / tot_time,
        "valid/n_ex": n_ex,
    }
    model.train()
    logger.info(f"validation finished in {int(tot_time)} seconds, " f"score: {val_acc * 100:.2f}")
    return val_log, results


@torch.no_grad()
def validate_generative_vqa(logger, model, val_loader, tokenizer, task_mask=None):
    model.eval()
    tot_score = 0
    n_ex = 0
    st = time()
    results = {}
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, batch in pbar:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                max_new_tokens=10,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        predictions = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        for qid, predicted_answer, ground_truth_answers in zip(batch["qids"], predictions, batch["answers"]):
            predicted_answer = normalize_answer(predicted_answer)
            ground_truth_counts = Counter(ground_truth_answers)

            acc_score = vqa_v2_score(ground_truth_counts.get(predicted_answer, 0))
            tot_score += acc_score
            results[qid] = {"answer": predicted_answer, "acc": acc_score}

        n_ex += len(batch["qids"])
        pbar.set_postfix({"Accuracy": f"{(tot_score / n_ex):.3f}"})

    tot_time = time() - st
    if isinstance(n_ex, torch.Tensor):
        n_ex = n_ex.item()
    if isinstance(tot_score, torch.Tensor):
        tot_score = tot_score.item()

    val_acc = tot_score / n_ex
    logger.info(f"Tested {n_ex} samples")
    val_log = {
        "valid/acc": val_acc,
        "valid/ex_per_s": n_ex / tot_time,
        "valid/n_ex": n_ex,
    }
    logger.info(f"validation finished in {int(tot_time)} seconds, " f"score: {val_acc * 100:.2f}")
    return val_log, results
