class CLStrategy:
    """Parent class for CL methods"""

    def __init__(self, reg_lambda=1.0, mask=None, scaler=None, **kwargs):
        """
        :param reg_lambda: weight of the regularization term
        """
        self.task_id = 0
        self.reg_lambda = reg_lambda
        self.mask = mask
        self.scaler = scaler
        if kwargs.get("opts") and kwargs["opts"].accumulate_grad_batches:
            self.update_freq = kwargs["opts"].accumulate_grad_batches
        else:
            self.update_freq = 1

    def update(self, model, **kwargs):
        """Update the method between tasks."""
        self.task_id += 1

    def update_after_new_task(self, **kwargs):
        """Update after initializing for the new task."""
        pass

    def update_after_backward(self, **kwargs):
        """Update after the backward pass."""
        pass

    def update_after_step(self, **kwargs):
        """Update after a gradient_step."""
        pass

    def compute_loss(self, model, loss, **kwargs):
        """Compute the loss."""
        raise NotImplementedError

    def replay(self, model, **kwargs):
        """Replay samples from memory."""
        loss = None
        n_ex = 0
        return loss, n_ex

    def _is_batch_after_step(self, batch_idx=0):
        """
        Update reference gradients. After step and before model.zero_grad().
        """
        return (batch_idx + 1) % self.update_freq == 0


class Naive(CLStrategy):
    """Naive Finetuning"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, loss, **kwargs):
        return loss
