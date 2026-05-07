# MSE + auxiliary loss (for models that return an aux_loss term, e.g. PERQH).
import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()

    def forward(
        self,
        pred: Tensor,
        true: Tensor,
        mask: Tensor | None = None,
        aux_loss: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        squared_error = residual ** 2
        num_eval = mask.sum()
        mse = squared_error.sum() / (num_eval if num_eval > 0 else 1)

        total = mse
        if aux_loss is not None:
            total = total + aux_loss

        return {
            "loss": total,
            "loss_mse": mse.detach(),
            "loss_aux": aux_loss.detach() if aux_loss is not None else torch.zeros_like(mse),
            "loss_reduction_none": squared_error,
        }
