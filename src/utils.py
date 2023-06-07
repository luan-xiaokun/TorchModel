"""Utilities for constructing training recipe and training routine"""
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f") -> None:
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        """Reset all values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1) -> None:
        """Update value"""
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        format_str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return format_str.format(**self.__dict__)


class ProgressMeter:
    """Display training progress and relevant meter info"""

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = "") -> None:
        self.batch_format_str = self._get_batch_format_str(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """Pretty print the progress information"""
        entries = [self.prefix + self.batch_format_str.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_format_str(self, num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def _check_args(name: str, class_obj: object, **kwargs: Any) -> None:
    arg_names = inspect.getfullargspec(class_obj).args[1:]
    if diff := set(kwargs.keys()).difference(arg_names):
        unexpected_args = ", ".join(diff)
        raise TypeError(f"unexpected arguments for {name}: {unexpected_args}")


def retrieve_last_layer(model: nn.Module):
    """Retrieve the last layer of a nn.Module, in a recursive way"""

    last_child = list(model.children())[-1]
    # the base case, e.g. Linear layer
    if len(list(last_child.children())) == 0:
        return last_child
    return retrieve_last_layer(last_child)


def set_parameter_weight_decay(
    model: nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[nn.Module]] = None,
):
    """Set weight decay to normal weights and normalization weights"""

    if not norm_classes:
        # pylint: disable=protected-access
        norm_classes = [
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.modules.instancenorm._InstanceNorm,
            nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }

    def _add_params(module: nn.Module):
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
            if norm_weight_decay is not None and isinstance(module, norm_classes):
                params["norm"].append(param)
            else:
                params["other"].append(param)
        for child_module in module.children():
            _add_params(child_module)

    _add_params(model)

    param_groups = []
    for key, val in params.items():
        if len(val) > 0:
            param_groups.append(
                {
                    "params": val,
                    "weight_decay": params_weight_decay[key],
                }
            )

    return param_groups


def get_train_val_dataloader(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loader"""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_user_defined_builder(path: str) -> Type[Any]:
    """Get user-defined builder"""
    # parser dataset_path
    match = re.search(r"((.*?\/)?(.*?)\.py)\/(\w+)", path)
    assert match is not None, "unexpected user-defined builder pattern"
    file_path = Path(match.group(1)).absolute()
    module_name = match.group(3)
    builder_name = match.group(4)

    # load module according to the path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return getattr(module, builder_name)
