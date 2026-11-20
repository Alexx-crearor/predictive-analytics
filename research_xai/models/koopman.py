import torch
from torch import nn

def _orth(w: torch.Tensor) -> torch.Tensor:
    """Trực chuẩn hóa ma trận bằng phân rã QR."""
    return torch.linalg.qr(w)[0]

class StrictStableKoopmanOperator(nn.Module):
    """Toán tử Koopman tuyến tính với ràng buộc ổn định."""

    def __init__(self, latent_dim: int, p_max: float = 0.99):
        super().__init__()

        # Khởi tạo hai ma trận dùng để xây dựng các ma trận trực giao
        self.U_raw = nn.Parameter(
            torch.randn(latent_dim, latent_dim)
        )
        self.V_raw = nn.Parameter(
            torch.randn(latent_dim, latent_dim)
        )

        # Khởi tạo các giá trị điều khiển độ co của toán tử
        self.S_raw = nn.Parameter(
            torch.randn(latent_dim)
        )

        # Giới hạn trên của singular values
        self.p_max = p_max

    def get_operator(self):
        """Xây dựng ma trận Koopman và lấy singular values."""

        # Trực chuẩn hóa hai ma trận tham số
        U = _orth(self.U_raw)
        V = _orth(self.V_raw)

        # Giới hạn singular values trong khoảng (0, ρ_max)
        singular_values = torch.sigmoid(self.S_raw) * self.p_max

        # Xây dựng toán tử Koopman theo phân rã ODO
        K = U @ torch.diag(singular_values) @ V.T

        return K, singular_values

    def step(self, z: torch.Tensor, K=None):
        """Thực hiện một bước chuyển trạng thái latent."""

        # Chỉ tính toán tử khi chưa được truyền từ bên ngoài
        if K is None:
            K, _ = self.get_operator()

        # Chuyển trạng thái bằng toán tử tuyến tính
        return z @ K.T

    def forward(self, z: torch.Tensor):
        # Xây dựng toán tử Koopman
        K, singular_values = self.get_operator()

        # Dự đoán trạng thái latent tiếp theo
        z_next = self.step(z, K)

        return z_next, K, singular_values


class AffineKoopmanOperator(StrictStableKoopmanOperator):
    """Toán tử Koopman affine với điểm cân bằng có thể học."""

    def __init__(self, latent_dim: int, p_max: float = 0.99):
        super().__init__(
            latent_dim=latent_dim,
            p_max=p_max
        )

        # Khởi tạo điểm cân bằng trong không gian latent
        self.mu = nn.Parameter(
            torch.zeros(latent_dim)
        )

    def step(self, z: torch.Tensor, K=None):
        """Chuyển trạng thái quanh điểm cân bằng mu."""

        # Tái sử dụng toán tử nếu đã được tính trước đó
        if K is None:
            K, _ = self.get_operator()

        # Dịch trạng thái về tâm, áp dụng Koopman rồi dịch ngược lại
        return self.mu + (z - self.mu) @ K.T
