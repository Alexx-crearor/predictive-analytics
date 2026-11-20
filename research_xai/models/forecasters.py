
import torch
from torch import nn

from .backbone import PatchTST_Backbone
from .koopman import (
    StrictStableKoopmanOperator,
    AffineKoopmanOperator,
)


class PatchTSTDirect(nn.Module):
    """M0: PatchTST baseline không sử dụng Koopman."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        patch_len: int,
        d_model: int = 96
    ):
        super().__init__()

        # Mã hóa chuỗi thời gian bằng PatchTST
        self.backbone = PatchTST_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model
        )

        # Dự báo trực tiếp toàn bộ horizon
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return self.fc(z)


class KoopmanDirect(nn.Module):
    """M1: DeepKoopFormer gốc với một lần biến đổi Koopman."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        patch_len: int,
        d_model: int = 96,
        p_max: float = 0.99
    ):
        super().__init__()

        # Trích xuất biểu diễn latent
        self.backbone = PatchTST_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model
        )

        # Toán tử Koopman ổn định của tác giả
        self.koop = StrictStableKoopmanOperator(
            latent_dim=d_model,
            p_max=p_max
        )

        # Dự báo toàn bộ horizon sau một lần biến đổi Koopman
        self.fc = nn.Linear(d_model, horizon * input_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_latents: bool = False
    ):
        z = self.backbone(x)

        # Áp dụng Koopman đúng một lần
        z_next, _, _ = self.koop(z)

        pred = self.fc(z_next)

        if return_latents:
            return pred, z, z_next

        return pred


class TemporalKoopman(nn.Module):
    """M2/M3/M4: Dự báo nhiều bước bằng Koopman rollout."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        patch_len: int,
        d_model: int = 96,
        p_max: float = 0.99,
        affine: bool = False
    ):
        super().__init__()

        if horizon < 1:
            raise ValueError("horizon phải lớn hơn hoặc bằng 1")

        self.horizon = horizon
        self.affine = affine

        # Sử dụng cùng kiến trúc PatchTST như baseline
        self.backbone = PatchTST_Backbone(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            d_model=d_model
        )

        # Chọn toán tử Koopman tuyến tính hoặc affine
        koopman_class = (
            AffineKoopmanOperator
            if affine
            else StrictStableKoopmanOperator
        )

        self.koop = koopman_class(
            latent_dim=d_model,
            p_max=p_max
        )

        # Decoder dùng chung để dự báo từng bước thời gian
        self.fc = nn.Linear(d_model, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False
    ):
        # Mã hóa cửa sổ quá khứ thành trạng thái ban đầu
        z = self.backbone(x)

        # Chỉ xây dựng ma trận Koopman một lần
        K, _ = self.koop.get_operator()

        predictions = []
        states = []

        for step in range(self.horizon):

            # Lưu trạng thái latent tại bước hiện tại
            states.append(z)

            # Dự báo một thời điểm bằng decoder dùng chung
            y_step = self.fc(z)
            predictions.append(y_step)

            # Chuyển trạng thái để dự báo thời điểm kế tiếp
            if step < self.horizon - 1:
                z = self.koop.step(z, K)

        # Ghép các dự báo theo chiều thời gian
        pred = torch.stack(predictions, dim=1)

        # Chuyển về thứ tự feature-major giống ablation cũ
        pred = pred.transpose(1, 2).reshape(x.size(0), -1)

        if return_states:
            # Trả thêm chuỗi latent để tính Alignment Loss
            states = torch.stack(states, dim=1)
            return pred, states

        return pred
