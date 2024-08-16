from torch.optim.lr_scheduler import LambdaLR


def warmup_linear_lr(learning_rate, step, warmup_steps, total_steps, min_lr=1e-8):
    """Linear warmup followed by linear decay."""
    if step < warmup_steps:
        return step * (learning_rate - min_lr) / warmup_steps + min_lr
    return max(min_lr, learning_rate * (total_steps - step) / (total_steps - warmup_steps))


def constant_lr(learning_rate, **kwargs):
    return learning_rate


def get_lr_sched(global_step, opts, total_steps=None, method="triangular"):
    if not total_steps:
        total_steps = opts.num_train_steps
    # learning rate scheduling
    if method == "triangular":
        if "warmup_steps" in vars(opts):
            warmup_steps = opts.warmup_steps
        else:
            warmup_steps = int(opts.warmup_perc * total_steps)

        lr_this_step = warmup_linear_lr(opts.learning_rate, global_step, warmup_steps, total_steps)
    elif method == "constant":
        lr_this_step = opts.learning_rate
    else:
        raise NotImplementedError(f"{method} not implemented")

    return lr_this_step


def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps, last_epoch=-1):
    """From huggingface.

    https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LRScheduler:
    def __init__(self, opts, total_steps=None, method="triangular"):
        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = opts.num_train_steps

        self.method = method
        self.method2update = {"constant": constant_lr, "triangular": warmup_linear_lr}
        if "warmup_steps" in vars(opts):
            self.warmup_steps = opts.warmup_steps
        else:
            self.warmup_steps = int(opts.warmup_perc * self.total_steps)
        self.learning_rate = opts.learning_rate

    def update(self, step):
        return self.method2update[self.method](
            learning_rate=self.learning_rate,
            step=step,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
        )
