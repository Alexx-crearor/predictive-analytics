
import numpy as np
import torch
from torch import nn

from .layers import PatchEmbed1D, SinCosPosEnc

class PatchTST_Backbone(nn.Module):
    """Mã hóa chuỗi thời gian thành biểu diễn latent bằng PatchTST."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_len: int,
        d_model: int = 48,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_ff: int = 96
    ):
        super().__init__()

        # Chuyển chuỗi thời gian thành các patch embedding
        self.patch = PatchEmbed1D(
            in_ch=input_dim,
            d_model=d_model,
            patch_len=patch_len,
            stride=patch_len
        )

        # Xác định số vị trí cần mã hóa, bao gồm CLS token
        n_patches = int(np.ceil(seq_len / patch_len))

        self.pos = SinCosPosEnc(
            d_model=d_model,
            max_len=n_patches + 1
        )

        # Khởi tạo một lớp Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True
        )

        # Xếp chồng các lớp Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Khởi tạo CLS token có thể học
        self.cls = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Trích xuất các patch embedding từ chuỗi đầu vào
        x = self.patch(x)

        # Mở rộng CLS token theo kích thước batch
        cls = self.cls.expand(x.size(0), -1, -1)

        # Ghép CLS token vào đầu chuỗi patch
        x = torch.cat([cls, x], dim=1)

        # Bổ sung thông tin vị trí cho các token
        x = self.pos(x)

        # Mã hóa các token bằng Transformer
        x = self.encoder(x)

        # Lấy CLS token làm biểu diễn latent của toàn bộ chuỗi
        return x[:, 0]
