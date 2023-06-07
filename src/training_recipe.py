"""Training recipe builder"""
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import prune

from dataset import get_dataset
from loss import get_builtin_loss_fn
from model import get_model
from training import train, validate
from optimizer import get_builtin_lr_scheduler, get_builtin_optimizer
from utils import get_train_val_dataloader, set_parameter_weight_decay


@dataclass
class TrainingRecipe:
    """A training recipe provides necessary information for the training routine"""

    model: nn.Module
    criterion: Callable[..., torch.Tensor]
    optimizer: optim.Optimizer
    train_dataset: Dataset
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    source: Optional[nn.Module] = None
    epoch_num: int = 20
    batch_size: int = 64
    training_method: str = "plain"
    pin_memory: bool = True
    num_workers: int = 4
    print_freq: int = 20
    save_checkpoint: bool = True
    checkpoint_path: str = "checkpoint.pt"
    device: str = "cuda"
    quantization_config: Dict[str, Any] = None

    def get_train_val_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        """Get training dataloader and validation dataloader"""
        train_loader, val_loader = get_train_val_dataloader(
            self.train_dataset, self.val_dataset, self.batch_size, self.num_workers, self.pin_memory
        )
        if self.val_dataset is None:
            val_loader = None
        return train_loader, val_loader

    def train(self) -> nn.Module:
        """Call the training procedure"""

        if self.training_method == "plain":
            return self.plain()

        if self.training_method in ["pruning", "prune"]:
            pass

        if self.training_method in ["quantization", "quantize", "quant"]:
            return self.quantization()

        if self.training_method in ["fine_tuning", "fine_tune", "finetuning", "finetune"]:
            return self.fine_tuning()

        if self.training_method in [
            "knowledge_distillation",
            "knowledgedistillation",
            "knowledge_distill",
            "knowledgedistill",
            "distillation",
            "distill",
            "kd",
        ]:
            return self.knowledge_distillation()

    def plain(self) -> nn.Module:
        """Plain training method implementation"""
        train_loader, val_loader = self.get_train_val_dataloader()
        print("start training...")
        model = train(
            self.model,
            train_loader,
            val_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            total_epoch_num=self.epoch_num,
            fine_tuning=False,
            device=self.device,
            print_freq=self.print_freq,
            resume_training=False,
            save_checkpoint=self.save_checkpoint,
            checkpoint_path=self.checkpoint_path,
        )

        if self.test_dataset:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            print("evaluating on test set...")
            validate(model, test_loader, self.criterion, self.device)

        return model

    def pruning(self) -> nn.Module:
        """Pruning implementation"""
        pass

    def fine_tuning(self) -> nn.Module:
        """Fine-tuning implementation"""
        pass

    def knowledge_distillation(self) -> nn.Module:
        """Knowledge distillation implementation"""
        pass

    def quantization(self) -> nn.Module:
        """Quantization implementation"""
        if self.quantization_config is None:
            self.quantization_config = {"dtype": "qint8", "static": False}
        static_quantization = self.quantization_config.get("static", False)
        # dynamic quantization
        if not static_quantization:
            dtype = self.quantization_config.get("dtype", "qint8")
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as exception:
                print(f"unsupported quantization type {dtype}")
                raise exception
            quantized = torch.quantization.quantize_dynamic(self.model, dtype=dtype)
        # static quantization
        else:
            if getattr(self.model, "fuse_model", None) is None:
                raise AttributeError(
                    "static quantization requires implementing `fuse_model` method"
                )
            model = deepcopy(self.model)

            def run_fn(model: nn.Module, dataset: Dataset):
                model.fuse_model()
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
                # expect that the dataloader yields (inputs, targets)
                with torch.inference_mode():
                    for inputs, _ in loader:
                        model(inputs)

            # use validation dataset for calibration
            # only use train dataset when there is no validation dataset
            quantized = torch.quantization.quantize(
                model,
                run_fn,
                run_args=(self.val_dataset if self.val_dataset else self.train_dataset,),
            )

        return quantized


def recipe_builder(config: Dict[str, Any]) -> TrainingRecipe:
    """Build a training recipe according to the given arguments"""
    # loss entry
    loss_setting: Dict[str, Any] = config.get("loss", {})
    loss_fn_name = loss_setting.get("class", "CrossEntropyLoss")
    loss_fn_args = {key: val for key, val in loss_setting.items() if key != "class"}
    criterion = get_builtin_loss_fn(loss_fn_name, **loss_fn_args)

    # model entry
    model_setting: Dict[str, Any] = config.get("model", {})
    model_class = model_setting.get("class", None)  # raise error if is None
    model_args = {key: val for key, val in model_setting.items() if key != "class"}
    model = get_model(model_class, **model_args)

    # optimizer entry
    optim_setting: Dict[str, Any] = config.get("optimizer", {})
    optim_name = optim_setting.get("class", "Adam")
    optim_args = {
        key: val
        for key, val in optim_setting.items()
        if key not in ["class", "weight_decay", "norm_weight_decay"]
    }
    weight_decay = optim_setting.get("weight_decay", 2e-5)
    norm_weight_decay = optim_setting.get("norm_weight_decay", None)
    parameters = set_parameter_weight_decay(model, weight_decay, norm_weight_decay)
    optimizer = get_builtin_optimizer(optim_name, parameters, **optim_args)

    # learning rate scheduler
    lr_scheduler_setting: Dict[str, Any] = config.get("lr_scheduler", {})
    lr_scheduler_name = lr_scheduler_setting.get("class", None)
    if lr_scheduler_name is not None:
        # TODO: multiple schedulers
        lr_scheduler_args = {
            key: val for key, val in lr_scheduler_setting.items() if key != "class"
        }
        lr_scheduler = get_builtin_lr_scheduler(lr_scheduler_name, optimizer, **lr_scheduler_args)
    else:
        lr_scheduler = None

    # dataset entry
    dataset_setting: Dict[str, Any] = config.get("dataset", {})
    dataset_class = dataset_setting.get("class", None)
    if dataset_class is None:
        raise AttributeError("dataset entry not found")
    dataset_args = {key: val for key, val in dataset_setting.items() if key != "class"}
    datasets = get_dataset(dataset_class, **dataset_args)
    datasets += (None,) * (3 - len(datasets))
    train_dataset, val_dataset, test_dataset = datasets

    # source entry
    source_setting: Dict[str, Any] = config.get("source", {})
    source_class = source_setting.get("class", None)
    if source_class is not None:
        source_args = {key: val for key, val in source_setting.items() if key != "class"}
        if "state_dict_path" not in source_args:
            raise AttributeError("source entry should specify `state_dict_path`")
        state_dict_path = source_args.pop("state_dict_path")
        source = get_model(source_class, **source_args)
        source_state_dict = torch.load(state_dict_path)
        if "state_dict" in source_state_dict:
            source_state_dict = source_state_dict["state_dict"]
        source.load_state_dict(source_state_dict)
    else:
        source = None

    # directly get configuration values
    training_setting: Dict[str, Any] = config.get("training", {})
    training_method = training_setting.get("class", "plain")
    epoch_num = training_setting.get("epoch_num", 20)
    batch_size = training_setting.get("batch_size", 64)
    pin_memory = training_setting.get("pin_memory", True)
    num_workers = training_setting.get("num_workers", 0)
    print_freq = training_setting.get("print_freq", 20)
    save_checkpoint = training_setting.get("save_checkpoint", True)
    checkpoint_path = training_setting.get("checkpoint_path", "checkpoint.pt")
    device = training_setting.get("device", "cuda")

    training_method = training_method.lower().replace(" ", "_").replace("-", "_")

    if not torch.cuda.is_available():
        pin_memory = False
        device = "cpu"

    if cpu_num := len(os.sched_getaffinity(0)) >= 8:
        num_workers = cpu_num // 2

    return TrainingRecipe(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epoch_num=epoch_num,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        source=source,
        training_method=training_method,
        pin_memory=pin_memory,
        num_workers=num_workers,
        print_freq=print_freq,
        save_checkpoint=save_checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )


if __name__ == "__main__":
    from recipe_parser.parser import parser

    with open("examples/example.tm", encoding="utf-8") as f:
        text = f.read()
    parse_result = parser.parse(text)
    recipe = recipe_builder(parse_result)
    recipe.train()
