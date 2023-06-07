"""Training routine and validation function"""
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import AverageMeter, ProgressMeter


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: Callable[..., torch.Tensor],
    optimizer: optim.Optimizer,
    epoch: int,
    total_epoch_num: int,
    fine_tuning: bool = False,
    device: str = "cuda",
    print_freq: int = 20,
) -> None:
    """Train the model for one epoch, support fine-tuning"""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc1],
        prefix=f"Epoch: [{epoch}/{total_epoch_num}]",
    )
    device = torch.device(device)
    if print_freq <= 0:
        print_freq = len(train_loader) + 1

    model.train()
    # when fine-tuning, normalization layer and dropout layer need to be freezed (in eval mode)
    if fine_tuning:
        model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # update data loading time
        data_time.update(time.time() - end)
        # push inputs and targets to device
        inputs: torch.Tensor = inputs.to(device, non_blocking=True)
        targets: torch.Tensor = targets.to(device, non_blocking=True)

        # compute outputs and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # record accuracy and loss
        # pylint: disable=unbalanced-tuple-unpacking
        (acc,) = accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        acc1.update(acc, inputs.size(0))
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (i + 1) % print_freq == 0:
            progress.display(i + 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: Callable[..., torch.Tensor],
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler],
    total_epoch_num: int,
    fine_tuning: bool = False,
    device: str = "cuda",
    print_freq: int = 20,
    resume_training: bool = False,
    save_checkpoint: bool = True,
    checkpoint_path: str = "checkpoint.pt",
) -> nn.Module:
    """Train a model for several epochs"""

    # TODO: add early stopping support

    device = torch.device(device)
    best_acc = 0.0
    start_epoch = 0

    if resume_training:
        print("==> resuming from checkpoint...")
        assert Path(checkpoint_path).exists(), "no checkpoint found"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])

    model.to(device)

    for epoch in range(start_epoch, start_epoch + total_epoch_num):
        train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch + 1,
            total_epoch_num=total_epoch_num,
            fine_tuning=fine_tuning,
            device=device,
            print_freq=print_freq,
        )
        if lr_scheduler is not None:
            lr_scheduler.step()

        if val_loader is not None:
            acc = validate(model, val_loader, criterion, device=device)
            if save_checkpoint:
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "state_dict": model.cpu().state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                }
                if lr_scheduler is not None:
                    checkpoint_state["scheduler"] = lr_scheduler.state_dict()
                torch.save(checkpoint_state, checkpoint_path)
                best_model_path = Path(checkpoint_path).parent / "best.pt"
                if is_best:
                    shutil.copyfile(checkpoint_path, best_model_path)
                model.to(device)

    return model


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Callable[..., torch.Tensor],
    device: str = "cuda",
    print_freq: int = 0,
) -> float:
    """Validate a model on validation / test dataset"""
    device = torch.device(device)
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, acc1], prefix="Test: ")
    device = torch.device(device)
    if print_freq <= 0:
        print_freq = len(val_loader) + 1

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            inputs: torch.Tensor = inputs.to(device, non_blocking=True)
            targets: torch.Tensor = targets.to(device, non_blocking=True)
            # compute outputs and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # record accuracy, loss, and time
            # pylint: disable=unbalanced-tuple-unpacking
            (acc,) = accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            acc1.update(acc, inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            # print info
            if (i + 1) % print_freq == 0:
                progress.display(i + 1)

    print(f" * Acc {acc1.avg:.3f}")

    return acc1.avg


def accuracy(outputs: torch.Tensor, targets: torch.Tensor, top_k=(1,)) -> List[float]:
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = targets.size(0)
        _, pred = outputs.topk(max_k, 1, largest=True, sorted=True)
        pred: torch.Tensor = pred.t()
        if targets.ndim == 2:
            targets = targets.max(dim=1)[1]
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
