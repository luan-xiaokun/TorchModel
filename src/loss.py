"""Construct loss function"""
import inspect
from typing import Any

import torch.nn as nn

from utils import _check_args


def collect_pytorch_builtin_losses():
    """Collect all pytorch built-in losses, such as CrossEntropyLoss, MSELoss,
    and return them as a dictionary, where the key is the lower-case name, and
    the value is the loss class object
    """
    losses = {}
    for name, obj in inspect.getmembers(nn.modules.loss):
        if inspect.isclass(obj) and not name.startswith("_") and "Loss" in name:
            losses[name.lower()] = obj
    return losses


def get_builtin_loss_fn(name: str, **loss_fn_args: Any) -> nn.Module:
    """Get built-in loss according to its name, the param does not need to match
    the built-in loss name exactly, a fuzzy match will be performed"""

    def _normalize_name(name: str) -> str:
        # remove all underscores "_", dashes "-", and whitespace " "
        name = name.replace("_", "").replace("-", "").replace(" ", "").replace("\t", "")
        # convert to lowercase
        name = name.lower()
        # we allow omitting the "Loss" or "loss" part in the input
        if "loss" not in name:
            name = name + "loss" if name != "nll2d" else "nllloss2d"
        return name

    # make sure that pytorch has such built-in loss function
    normalized_name = _normalize_name(name)
    if normalized_name not in BUILTIN_LOSSES:
        raise KeyError(f"{name} is not a pytorch built-in loss")

    # check arguments
    loss_fn_class = BUILTIN_LOSSES[normalized_name]
    _check_args(name, loss_fn_class, **loss_fn_args)

    return loss_fn_class(**loss_fn_args)


BUILTIN_LOSSES = collect_pytorch_builtin_losses()
