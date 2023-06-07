"""Construct model"""
from typing import Any

import torch.nn as nn

from utils import _check_args, get_user_defined_builder


def get_model(model_path: str, **model_args: Any) -> nn.Module:
    """User-defined model builder"""
    model_builder = get_user_defined_builder(model_path)

    _check_args(model_path, model_builder, **model_args)

    return model_builder(**model_args)


if __name__ == "__main__":
    get_model("examples/mynet.py/MobileNetV2", num_classes=10)
