import torch
from torch import nn
from torch.nn import functional as F


class ForecastingLoss(nn.Module):
    """Tính sai số MSE giữa dự báo và giá trị thực."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        if predictions.shape != targets.shape:
            raise ValueError("Kích thước dự báo và nhãn không khớp")

        return F.mse_loss(predictions, targets)


class TemporalAlignmentLoss(nn.Module):
    """Căn chỉnh các trạng thái Koopman với latent của cửa sổ tương lai."""

    def __init__(self, max_steps: int = 2):
        super().__init__()

        if max_steps < 1:
            raise ValueError("max_steps phải lớn hơn hoặc bằng 1")

        self.max_steps = max_steps

    def forward(
        self,
        states: torch.Tensor,
        past: torch.Tensor,
        future: torch.Tensor,
        teacher_encoder: nn.Module,
        steps=None
    ) -> torch.Tensor:

        batch_size, horizon, latent_dim = states.shape
        _, seq_len, input_dim = past.shape

        if future.shape != (batch_size, input_dim * horizon):
            raise ValueError("Kích thước future không hợp lệ")

        if horizon < 2:
            return states.sum() * 0.0

        # Chọn các bước chuyển trạng thái cần căn chỉnh
        if steps is None:
            n_steps = min(self.max_steps, horizon - 1)

            steps = (
                torch.randperm(horizon - 1)[:n_steps] + 1
            ).tolist()
        else:
            steps = [int(step) for step in steps]

        if not steps or any(step < 1 or step >= horizon for step in steps):
            raise ValueError("Bước alignment phải nằm trong [1, horizon - 1]")

        # Khôi phục thứ tự thời gian từ nhãn feature-major
        future_sequence = future.reshape(
            batch_size, input_dim, horizon
        ).transpose(1, 2)

        # Ghép quá khứ và tương lai để tạo các cửa sổ dịch thời gian
        full_sequence = torch.cat(
            [past, future_sequence],
            dim=1
        )

        # Lấy các trạng thái Koopman cần kiểm tra
        predicted_states = states[:, steps, :]

        # Teacher tạo latent mục tiêu, không nhận gradient
        with torch.no_grad():

            shifted_windows = torch.stack(
                [
                    full_sequence[:, step:step + seq_len, :]
                    for step in steps
                ],
                dim=1
            )

            # Gộp các cửa sổ để mã hóa trong một lần forward
            shifted_windows = shifted_windows.reshape(
                batch_size * len(steps),
                seq_len,
                input_dim
            )

            target_states = teacher_encoder(shifted_windows)

            target_states = target_states.reshape(
                batch_size,
                len(steps),
                latent_dim
            )

        # Đưa latent dự đoán đến gần latent của cửa sổ tương lai
        return F.mse_loss(predicted_states, target_states)
