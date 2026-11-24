
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import hashlib
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import ForecastingLoss, TemporalAlignmentLoss


@dataclass
class TrainConfig:
    epochs: int = 4000
    batch_size: int = 32
    lr: float = 3e-4
    lambda_align: float = 0.1
    ema_decay: float = 0.99
    max_align_steps: int = 2
    val_every: int = 100
    checkpoint_every: int = 200
    grad_clip: float = 1.0
    early_stopping_patience: int = 800
    early_stopping_min_delta: float = 1e-5
    seed: int = 7


def set_seed(seed: int):
    """Cá»‘ Ä‘á»‹nh seed Ä‘á»ƒ há»— trá»£ tÃ¡i láº­p thÃ­ nghiá»‡m."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state():
    """LÆ°u tráº¡ng thÃ¡i ngáº«u nhiÃªn Ä‘á»ƒ resume training."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    """KhÃ´i phá»¥c tráº¡ng thÃ¡i ngáº«u nhiÃªn tá»« checkpoint."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def copy_weights_to_cpu(model):
    """Sao chÃ©p trá»ng sá»‘ tá»‘t nháº¥t vá» CPU."""

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_checkpoint(path, payload):
    """LÆ°u checkpoint an toÃ n qua file táº¡m."""

    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def calculate_data_hash(*tensors):
    """Táº¡o mÃ£ kiá»ƒm tra dá»¯ liá»‡u Ä‘á»ƒ trÃ¡nh resume nháº§m."""

    digest = hashlib.sha256()

    for tensor in tensors:
        array = tensor.detach().contiguous().cpu().numpy()

        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())

    return digest.hexdigest()


@torch.no_grad()
def update_ema(teacher, student, decay):
    """Cáº­p nháº­t teacher báº±ng trung bÃ¬nh trÆ°á»£t trá»ng sá»‘ student."""

    student_parameters = dict(student.named_parameters())

    for name, parameter in teacher.named_parameters():
        parameter.mul_(decay).add_(
            student_parameters[name],
            alpha=1.0 - decay
        )

    # Äá»“ng bá»™ cÃ¡c buffer khÃ´ng pháº£i tham sá»‘ há»c
    student_buffers = dict(student.named_buffers())

    for name, buffer in teacher.named_buffers():
        buffer.copy_(student_buffers[name])


@torch.no_grad()
def evaluate_mse(model, loader, device):
    """ÄÃ¡nh giÃ¡ MSE trÃªn táº­p validation."""

    model.eval()

    total_error = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)

        loss = torch.nn.functional.mse_loss(
            predictions, targets
        )

        total_error += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return total_error / total_samples


def train_model(
    model,
    variant,
    x_train,
    y_train,
    x_val,
    y_val,
    run_dir,
    config=None,
    resume=True
):
    """Huáº¥n luyá»‡n má»™t cáº¥u hÃ¬nh cá»§a M0, M1, M2, M3 hoáº·c M4."""

    if config is None:
        config = TrainConfig()

    if variant not in {"M0", "M1", "M2", "M3", "M4", "LSTM", "DLinear"}:
        raise ValueError("Biáº¿n thá»ƒ mÃ´ hÃ¬nh khÃ´ng há»£p lá»‡")

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs vÃ  batch_size pháº£i dÆ°Æ¡ng")

    if not 0 <= config.ema_decay < 1:
        raise ValueError("ema_decay pháº£i náº±m trong [0, 1)")

    if config.val_every < 1 or config.checkpoint_every < 1:
        raise ValueError("Khoáº£ng lÆ°u vÃ  Ä‘Ã¡nh giÃ¡ pháº£i dÆ°Æ¡ng")

    if config.lambda_align < 0 or config.grad_clip <= 0:
        raise ValueError("Cáº¥u hÃ¬nh loss hoáº·c gradient khÃ´ng há»£p lá»‡")

    # ÄÆ°a dá»¯ liá»‡u vá» CPU Ä‘á»ƒ trÃ¡nh chiáº¿m toÃ n bá»™ VRAM
    x_train, y_train, x_val, y_val = [
        tensor.detach().to(
            "cpu", dtype=torch.float32
        ).contiguous()
        for tensor in (x_train, y_train, x_val, y_val)
    ]

    if x_train.ndim != 3 or x_val.ndim != 3:
        raise ValueError("Input pháº£i cÃ³ dáº¡ng (B, L, C)")

    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError("NhÃ£n pháº£i cÃ³ dáº¡ng (B, C * H)")

    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Train vÃ  validation khÃ´ng Ä‘Æ°á»£c rá»—ng")

    if x_train.shape[1:] != x_val.shape[1:]:
        raise ValueError("Shape input khÃ´ng khá»›p")

    if (
        y_train.shape[1] != y_val.shape[1]
        or y_train.shape[1] % x_train.shape[2] != 0
    ):
        raise ValueError("Shape nhÃ£n khÃ´ng há»£p lá»‡")

    for tensor in (x_train, y_train, x_val, y_val):
        if not torch.isfinite(tensor).all().item():
            raise ValueError("Dá»¯ liá»‡u chá»©a NaN hoáº·c Inf")

    # Kiá»ƒm tra dá»¯ liá»‡u cÃ³ thay Ä‘á»•i giá»¯a cÃ¡c láº§n cháº¡y khÃ´ng
    data_hash = calculate_data_hash(
        x_train, y_train, x_val, y_val
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_path = run_dir / "checkpoint_latest.pt"
    best_path = run_dir / "checkpoint_best.pt"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    # Khá»Ÿi táº¡o optimizer cho toÃ n bá»™ tham sá»‘ cá»§a mÃ´ hÃ¬nh
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr
    )

    forecast_loss_fn = ForecastingLoss()
    alignment_loss_fn = TemporalAlignmentLoss(
        max_steps=config.max_align_steps
    )

    # Chá»‰ M3 vÃ  M4 sá»­ dá»¥ng Alignment Loss
    use_alignment = variant in {"M3", "M4"}

    teacher = None

    if use_alignment:
        # Teacher ban Ä‘áº§u lÃ  báº£n sao cá»§a backbone
        teacher = deepcopy(model.backbone).to(device)
        teacher.requires_grad_(False)
        teacher.eval()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    start_epoch = 0
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_weights = None

    # KhÃ´i phá»¥c quÃ¡ trÃ¬nh huáº¥n luyá»‡n náº¿u Ä‘Ã£ cÃ³ checkpoint
    if resume and latest_path.exists():
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False
        )

        if (
            checkpoint["variant"] != variant
            or checkpoint["config"] != asdict(config)
        ):
            raise ValueError("Checkpoint khÃ´ng khá»›p cáº¥u hÃ¬nh")

        if checkpoint["data_hash"] != data_hash:
            raise ValueError("Dá»¯ liá»‡u khÃ´ng khá»›p checkpoint")

        model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if teacher is not None:
            teacher.load_state_dict(checkpoint["teacher"])

        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        best_val = checkpoint["best_val"]
        best_epoch = checkpoint["best_epoch"]
        best_weights = checkpoint["best_weights"]

        # KhÃ´i phá»¥c thá»© tá»± láº¥y máº«u vÃ  cÃ¡c phÃ©p ngáº«u nhiÃªn
        restore_rng_state(checkpoint["rng"])

        print(f"Resume tá»« epoch {start_epoch}")

    else:
        set_seed(config.seed)

    def save_latest(epoch):
        """LÆ°u Ä‘áº§y Ä‘á»§ tráº¡ng thÃ¡i Ä‘á»ƒ tiáº¿p tá»¥c huáº¥n luyá»‡n."""

        payload = {
            "epoch": epoch,
            "variant": variant,
            "config": asdict(config),
            "data_hash": data_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "teacher": (
                teacher.state_dict()
                if teacher is not None else None
            ),
            "history": history,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_weights": best_weights,
            "rng": get_rng_state()
        }

        save_checkpoint(latest_path, payload)
    # Theo dÃµi sá»‘ epoch thá»±c táº¿ vÃ  tráº¡ng thÃ¡i dá»«ng sá»›m
    last_epoch = start_epoch
    early_stopped = False
    # SYNC_RUNTIME_225_V1: gioi han thoi gian chi khi co bien moi truong.
    stop_deadline = float(os.environ.get("RESEARCH_XAI_STOP_DEADLINE", "0"))
    time_limited = False
    
    for epoch in range(start_epoch, config.epochs):
        model.train()

        total_loss_sum = 0.0
        forecast_sum = 0.0
        alignment_sum = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Láº¥y latent states khi cáº§n tÃ­nh Alignment Loss
            if use_alignment:
                predictions, states = model(
                    inputs,
                    return_states=True
                )
            else:
                predictions = model(inputs)

            loss_forecast = forecast_loss_fn(
                predictions, targets
            )

            if use_alignment:
                loss_alignment = alignment_loss_fn(
                    states=states,
                    past=inputs,
                    future=targets,
                    teacher_encoder=teacher
                )
            else:
                loss_alignment = torch.zeros(
                    (), device=device
                )

            # Káº¿t há»£p forecasting loss vÃ  alignment loss
            loss = (
                loss_forecast
                + config.lambda_align * loss_alignment
            )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Loss khÃ´ng há»£p lá»‡ táº¡i epoch {epoch + 1}"
                )

            # Lan truyá»n gradient vÃ  cáº­p nháº­t tham sá»‘
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip
            )

            optimizer.step()

            # Cáº­p nháº­t EMA teacher sau má»—i láº§n optimizer cáº­p nháº­t
            if teacher is not None:
                update_ema(
                    teacher,
                    model.backbone,
                    config.ema_decay
                )

            batch_size = inputs.size(0)

            total_loss_sum += loss.item() * batch_size
            forecast_sum += (
                loss_forecast.item() * batch_size
            )
            alignment_sum += (
                loss_alignment.item() * batch_size
            )


        current_epoch = epoch + 1
        last_epoch = current_epoch

        val_mse = None
        stop_training = False

        # ÄÃ¡nh giÃ¡ mÃ´ hÃ¬nh theo chu ká»³ validation
        if (
            current_epoch % config.val_every == 0
            or current_epoch == config.epochs
        ):
            val_mse = evaluate_mse(
                model, val_loader, device
            )

            # Kiá»ƒm tra validation cÃ³ cáº£i thiá»‡n Ä‘á»§ má»©c yÃªu cáº§u khÃ´ng
            if val_mse < best_val - config.early_stopping_min_delta:
                best_val = val_mse
                best_epoch = current_epoch
                best_weights = copy_weights_to_cpu(model)

                # LÆ°u ngay checkpoint tá»‘t nháº¥t Ä‘á»ƒ trÃ¡nh máº¥t khi ngáº¯t
                save_checkpoint(
                    best_path,
                    {
                        "model": best_weights,
                        "epoch": best_epoch,
                        "val_mse": best_val,
                        "variant": variant,
                        "config": asdict(config),
                        "data_hash": data_hash
                    }
                )

            # Dá»«ng khi validation khÃ´ng cáº£i thiá»‡n trong thá»i gian dÃ i
            elif (
                current_epoch - best_epoch
                >= config.early_stopping_patience
                and current_epoch < config.epochs
            ):
                stop_training = True

        # Ghi lá»‹ch sá»­ huáº¥n luyá»‡n Ä‘á»ƒ phÃ¢n tÃ­ch há»™i tá»¥
        history.append({
            "epoch": current_epoch,
            "train_forecast_mse": forecast_sum / len(x_train),
            "train_alignment_mse": alignment_sum / len(x_train),
            "val_mse": val_mse,
            "best_val_mse": best_val,
        })

        # LÆ°u tráº¡ng thÃ¡i Ä‘á»‹nh ká»³ hoáº·c ngay khi Early Stopping kÃ­ch hoáº¡t
        if (
            current_epoch % config.checkpoint_every == 0
            or current_epoch == config.epochs
            or stop_training
        ):
            save_latest(current_epoch)

            print(
                f"[{variant}] Epoch "
                f"{current_epoch}/{config.epochs} | "
                f"Forecast MSE: "
                f"{forecast_sum / len(x_train):.6f} | "
                f"Val MSE: {val_mse if val_mse is not None else float('nan'):.6f} | "
                f"Best Val MSE: {best_val:.6f}",
                flush=True
            )

        # Káº¿t thÃºc training nhÆ°ng váº«n giá»¯ láº¡i checkpoint tá»‘t nháº¥t
        if stop_training:
            early_stopped = True

            print(
                f"[EARLY STOP] {variant} | "
                f"Epoch: {current_epoch} | "
                f"Best epoch: {best_epoch} | "
                f"Best Val MSE: {best_val:.6f}",
                flush=True
            )

            break


        # Luon luu trang thai neu phien Kaggle den han.
        if (stop_deadline > 0 and current_epoch < config.epochs
                and time.time() >= stop_deadline):
            save_latest(current_epoch)
            time_limited = True
            print(f"[TIME LIMIT] {variant}: checkpoint epoch {current_epoch}", flush=True)
            break

    # LÆ°u mÃ´ hÃ¬nh cÃ³ validation MSE tá»‘t nháº¥t
    if best_weights is not None:
        save_checkpoint(
            best_path,
            {
                "model": best_weights,
                "epoch": best_epoch,
                "val_mse": best_val,
                "variant": variant,
                "config": asdict(config),
                "data_hash": data_hash
            }
        )

    return {
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "early_stopped": early_stopped,
        "epochs_saved": config.epochs - last_epoch,
        "run_dir": str(run_dir),
        "time_limited": time_limited
    }

