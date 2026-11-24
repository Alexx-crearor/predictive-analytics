import torch
from research_xai.models.baselines import LSTMForecaster, DLinearForecaster


def test_shapes_and_gradients():
    for model in (LSTMForecaster(3, 100, 8), DLinearForecaster(3, 100, 8)):
        x = torch.randn(4, 100, 3, requires_grad=True)
        y = model(x)
        assert y.shape == (4, 24)
        assert torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dlinear_output_not_scalar_repeat():
    model = DLinearForecaster(3, 100, 8)
    y = model(torch.randn(4, 100, 3)).reshape(4, 3, 8)
    assert not torch.allclose(y[..., 0], y[..., 1]), "DLinear accidentally repeats one value across horizons"


def test_feature_major_order_dlinear():
    model = DLinearForecaster(2, 100, 8)
    x = torch.randn(1, 100, 2)
    trend = model.moving_avg(x)
    expected = (model.linear_seasonal((x-trend).transpose(1, 2)) +
                model.linear_trend(trend.transpose(1, 2)))
    torch.testing.assert_close(model(x).reshape(1, 2, 8), expected)
