import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class NoamLR(LRScheduler):
    def __init__(self, optimizer, model_size, factor, warmup, last_epoch=-1):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        rate = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return [rate for _ in self.base_lrs]


class InverseSqrtLR(LRScheduler):
    def __init__(self, optimizer, ref_lr: float, ref_steps: int, warmup_steps: int, last_epoch=-1):
        self.ref_lr = ref_lr
        self.ref_steps = ref_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        lr = self.ref_lr
        if self.ref_steps > 0:
            lr /= np.sqrt(max(step / self.ref_steps, 1))
        if self.warmup_steps > 0:
            lr *= min(step / self.warmup_steps, 1)

        return [lr for _ in self.base_lrs]
