"""Construct dataset"""
import inspect
from pathlib import Path
from typing import Any, Optional, Tuple, Type

import torch.nn as nn
from torch.utils.data import Dataset, random_split
import torchvision.datasets
import torchaudio.datasets


from utils import get_user_defined_builder


def collect_pytorch_builtin_datasets():
    """Collect all pytorch built-in datasets, such as MNIST, CIFAR10,
    and return them as a dictionary, where the key is the lower-case name, and
    the value is the dataset object
    """
    vision_datasets = {}
    for name, obj in inspect.getmembers(torchvision.datasets):
        if (
            inspect.isclass(obj)
            and issubclass(obj, Dataset)
            and obj is not torchvision.datasets.VisionDataset
            and obj is not torchvision.datasets.ImageFolder
            and obj is not torchvision.datasets.DatasetFolder
            # maybe not including LSUNClass?
        ):
            vision_datasets[name.lower()] = obj
    audio_datasets = {}
    for name, obj in inspect.getmembers(torchaudio.datasets):
        if inspect.isclass(obj) and issubclass(obj, Dataset):
            audio_datasets[name.lower()] = obj

    return {**vision_datasets, **audio_datasets}


def get_builtin_dataset_builder(name: str) -> Type[Dataset]:
    """Get built-in dataset according to its name, the param does not need to
    match the built-in dataset name exactly, a fuzzy match will be performed"""

    def _normalize_name(name: str) -> str:
        # remove all dashes "-" and whitespace " "
        name = name.replace("-", "").replace(" ", "").replace("\t", "")
        # convert to lowercase
        name = name.lower()
        return name

    # make sure that pytorch has such built-in dataset
    normalized_name = _normalize_name(name)
    if normalized_name not in BUILTIN_DATASETS:
        raise KeyError(f"{name} is not a pytorch built-in dataset")

    return BUILTIN_DATASETS[normalized_name]


def get_dataset(dataset_class: str, **dataset_args: Any) -> Tuple[Dataset]:
    """Get dataset according to the parsed dataset_class"""
    builtin_dataset = False
    # indicating that this is a user-defined dataset
    if ".py" in dataset_class:
        dataset_builder = get_user_defined_builder(dataset_class)
    # do we need to support TorchModel dataset (code reuse)?
    else:
        dataset_builder = get_builtin_dataset_builder(dataset_class)
        builtin_dataset = True

    if builtin_dataset:  # special case for built-in datasets
        pass

    arg_names = inspect.getfullargspec(dataset_builder.__init__).args[1:]

    # deal with download
    if "root" in dataset_args:
        root_path = Path(dataset_args["root"])
        if root_path.is_dir() and not any(root_path.iterdir()) and "download" in arg_names:
            dataset_args["download"] = True

    # deal with transform
    transform = create_transform(dataset_args.get("transform"))
    target_transform = create_transform(dataset_args.get("target_transform"))
    dataset_args["transform"] = transform
    dataset_args["target_transform"] = target_transform

    # check arguments
    # case 1: user gives `test_ratio`, indicating that this is a user-defined dataset
    #         and that there is no `split` or `train` argument
    if "test_ratio" in dataset_args:
        test_ratio = dataset_args["test_ratio"]
        val_ratio = dataset_args.get("val_ratio", 0.0)
        train_ratio = 1.0 - val_ratio - test_ratio
        dataset_args.pop("test_ratio", None)
        dataset_args.pop("val_ratio", None)
        dataset = dataset_builder(**dataset_args)
        if val_ratio == 0.0:
            lengths = [train_ratio, test_ratio]
        else:
            lengths = [train_ratio, val_ratio, test_ratio]
        return tuple(random_split(dataset, lengths))

    # there is no `test_ratio`, there must be a `split` or `train` argument
    # case 2: the dataset has `train` argument, indicating that there is no provided
    #         val dataset, user may provide `val_ratio`
    if "train" in arg_names:
        if "val_ratio" in dataset_args:
            val_ratio = dataset_args["val_ratio"]
            train_ratio = 1.0 - val_ratio
            del dataset_args["val_ratio"]
            train_val_dataset = dataset_builder(train=True, **dataset_args)
            train_dataset, val_dataset = random_split(train_val_dataset, [train_ratio, val_ratio])
            test_dataset = dataset_builder(train=False, **dataset_args)
            return train_dataset, val_dataset, test_dataset

        train_dataset = dataset_builder(train=True, **dataset_args)
        test_dataset = dataset_builder(train=False, **dataset_args)
        return train_dataset, None, test_dataset

    # case 3: the dataset has `split` argument, or `subset` argument, but no `train`
    #         argument, and there is no `test_ratio` provided
    if (param := "split") in arg_names or (param := "subset") in arg_names:
        if "val_ratio" in dataset_args:
            val_ratio = dataset_args["val_ratio"]
            train_ratio = 1.0 - val_ratio
            del dataset_args["val_ratio"]
            train_val_dataset = dataset_builder(**{param: "train", **dataset_args})
            train_dataset, val_dataset = random_split(train_val_dataset, [train_ratio, val_ratio])
            test_dataset = dataset_builder(**{param: "test", **dataset_args})
            return train_dataset, val_dataset, test_dataset

        train_dataset = dataset_builder(**{param: "train", **dataset_args})
        test_dataset = dataset_builder(**{param: "test", **dataset_args})
        try:
            val_dataset = dataset_builder(**{param: "val", **dataset_args})
            return train_dataset, val_dataset, test_dataset
        except ValueError:
            return train_dataset, None, test_dataset

    # no `test_ratio` provided, no `train`, `split` or `subset` argument
    # case 4: the user only want to use the dataset for training, no val or test
    train_dataset = dataset_builder(**dataset_args)
    return (train_dataset, None, None)


def create_transform(transform_str: Optional[str]) -> nn.Module:
    """Create transform object based on given long string"""
    if transform_str is None:
        return None
    return get_user_defined_builder(transform_str)


BUILTIN_DATASETS = collect_pytorch_builtin_datasets()


if __name__ == "__main__":
    count = 1
    for d_name, builder in BUILTIN_DATASETS.items():
        args = inspect.getfullargspec(builder.__init__).args[1:]
        if "split" not in args and "train" not in args:
            print(count, d_name)
            print(args)
            count += 1
