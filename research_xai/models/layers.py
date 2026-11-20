import numpy as np
import torch
from torch import nn

class SinCosPosEnc(nn.Module):
    """Bổ sung thông tin vị trí cho các token đầu vào Transformer."""

    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()

        # Tạo chỉ số vị trí cho các token
        pos = torch.arange(max_len).float().unsqueeze(1)

        # Tính tần số tương ứng với từng chiều embedding
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10_000.0) / d_model)
        )

        # Khởi tạo ma trận positional encoding
        pe = torch.zeros(max_len, d_model)

        # Áp dụng hàm sin cho chiều chẵn và cos cho chiều lẻ
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Lưu positional encoding dưới dạng buffer không học
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cộng thông tin vị trí vào các token
        return x + self.pe[:x.size(1)]

class PatchEmbed1D(nn.Module):
    """Chuyển chuỗi thời gian đa biến thành các patch embedding."""

    def __init__(
        self,
        in_ch: int,
        d_model: int,
        patch_len: int,
        stride: int
    ):
        super().__init__()

        # Dùng Conv1d để trích xuất và mã hóa các patch
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chuyển dữ liệu sang định dạng đầu vào của Conv1d
        x = x.permute(0, 2, 1)

        # Trích xuất các patch embedding
        x = self.conv(x)

        # Đưa chiều patch về đúng vị trí để dùng cho Transformer
        return x.permute(0, 2, 1)