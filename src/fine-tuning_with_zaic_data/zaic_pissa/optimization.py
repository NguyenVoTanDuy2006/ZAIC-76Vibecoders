import math
import torch


class OptimizerController:
    def __init__(self, optimizer, scheduler, scaler, clip, runtime):
        self.optimizer, self.scheduler, self.scaler, self.clip, self.runtime = (
            optimizer,
            scheduler,
            scaler,
            clip,
            runtime,
        )

    def backward(self, loss, group_size):
        if group_size < 1:
            raise ValueError("invalid accumulation group size")
        loss = loss / group_size
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        parameters = [p for p in model.parameters() if p.requires_grad]
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters
        )
        finite = self.runtime.agree(finite)
        norm = (
            float(torch.nn.utils.clip_grad_norm_(parameters, self.clip))
            if finite
            else float("inf")
        )
        finite = self.runtime.agree(finite and math.isfinite(norm))
        old_scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
        if finite:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        elif self.scaler is not None:
            self.scaler.update(new_scale=max(1.0, old_scale / 2))
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "successful": finite,
            "skipped": not finite,
            "grad_norm": norm if math.isfinite(norm) else None,
            "scale": float(self.scaler.get_scale()) if self.scaler is not None else 1.0,
        }
