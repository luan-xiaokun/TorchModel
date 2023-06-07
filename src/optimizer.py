"""Construct optimizer and learning rate scheduler"""
import inspect
from typing import Any, Iterable

import torch.optim as optim

from utils import _check_args


def collect_pytorch_builtin_optimizers():
    """Collect all pytorch built-in optimizers, such as SGD, Adam,
    and return them as a dictionary, where the key is the lower-case name, and
    the value is the optimizer class object
    """
    optimizers = {}
    for name, obj in inspect.getmembers(optim):
        if inspect.isclass(obj) and issubclass(obj, optim.Optimizer) and obj is not optim.Optimizer:
            optimizers[name.lower()] = obj
    return optimizers


def collect_pytorch_builtin_lr_schedulers():
    """Collect all pytorch built-in learning rate schedulers, such as StepLR,
    and return them as a dictionary, where the key is the lower-case name, and
    the value is the lr scheduler class object
    """
    lr_schedulers = {}
    for name, obj in inspect.getmembers(optim.lr_scheduler):
        if (
            inspect.isclass(obj)
            and issubclass(obj, optim.lr_scheduler.LRScheduler)
            and obj is not optim.lr_scheduler.LRScheduler
            and not name.startswith("_")
        ):
            lr_schedulers[name.lower()] = obj
        elif name == "ReduceLROnPlateau":
            lr_schedulers[name.lower()] = obj
    return lr_schedulers


def get_builtin_optimizer(
    name: str, parameters: Iterable, **optimizer_args: Any
) -> optim.Optimizer:
    """Get built-in optimizer according to its name, the param does not need to match
    the built-in optimizer name exactly, a fuzzy match will be performed"""

    def _normalize_name(name: str) -> str:
        # remove all underscores "_", dashes "-", and whitespace " "
        name = name.replace("_", "").replace("-", "").replace(" ", "").replace("\t", "")
        # convert to lowercase
        name = name.lower()
        return name

    # make sure that pytorch has such built-in optimizer
    normalized_name = _normalize_name(name)
    if normalized_name not in BUILTIN_OPTIMIZERS:
        raise KeyError(f"{name} is not a pytorch built-in optimizer")

    # check arguments
    optimizer_class = BUILTIN_OPTIMIZERS[normalized_name]
    _check_args(name, optimizer_class, **optimizer_args)

    return optimizer_class(parameters, **optimizer_args)


def get_builtin_lr_scheduler(
    name: str, optimizer: optim.Optimizer, **lr_scheduler_args: Any
) -> object:
    """Get built-in learning rate scheduler according to its name, the param does not
    need to match the built-in scheduler name exactly, a fuzzy match will be performed"""

    def _normalize_name(name: str) -> str:
        # remove all underscores "_", dashes "-", and whitespace " "
        name = name.replace("_", "").replace("-", "").replace(" ", "").replace("\t", "")
        # convert to lowercase
        name = name.lower()
        # we allow omitting the "Loss" or "loss" part in the input
        if "lr" not in name and name != "chainedscheduler":
            name = name + "lr" if name != "reduceonplateau" else "reducelronplateau"
        return name

    # make sure that pytorch has such built-in lr scheduler
    normalized_name = _normalize_name(name)
    if normalized_name not in BUILTIN_LR_SCHEDULERS:
        raise KeyError(f"{name} is not a pytorch built-in lr scheduler")

    # check arguments
    lr_scheduler_class = BUILTIN_LR_SCHEDULERS[normalized_name]
    _check_args(name, lr_scheduler_class, **lr_scheduler_args)

    # TODO: deal with LambdaLR, SequentialLR, ChainedScheduler

    return lr_scheduler_class(optimizer, **lr_scheduler_args)


BUILTIN_OPTIMIZERS = collect_pytorch_builtin_optimizers()
BUILTIN_LR_SCHEDULERS = collect_pytorch_builtin_lr_schedulers()
