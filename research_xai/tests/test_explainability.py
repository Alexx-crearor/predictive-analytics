"""Unit tests XAI: test công thức bằng mô hình tuyến tính có đáp án giải tích."""
import unittest

import torch
from torch import nn

from research_xai.explainability.integrated_gradients import (
    integrated_gradients, target_prediction,
)
from research_xai.explainability.temporal_occlusion import temporal_occlusion
from research_xai.explainability.koopman_attribution import koopman_diagnostics
from research_xai.scripts.run_xai import select_indices


class LinearForecast(nn.Module):
    def __init__(self):
        super().__init__()
        # Input (B,3,2). Output feature-major (B,C=2,H=2)
        self.register_buffer("weights", torch.tensor([
            [1., 2.], [3., 4.], [5., 6.],
        ]))

    def forward(self, x):
        scalar = (x * self.weights).sum(dim=(1, 2))
        zeros = torch.zeros_like(scalar)
        # output channel0 lead1 là scalar
        return torch.stack((zeros, scalar, zeros, zeros), dim=1)


class ToyBackbone(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class ToyKoop(nn.Module):
    def __init__(self, affine=False):
        super().__init__()
        self.register_buffer("K", torch.tensor([[0.5, 0.], [0., 0.7]]))
        if affine:
            self.register_buffer("mu", torch.tensor([0.2, 0.3]))

    def get_operator(self):
        return self.K, torch.tensor([0.5, 0.7])

    def step(self, z, K=None):
        K = self.K if K is None else K
        if hasattr(self, "mu"):
            return self.mu + (z - self.mu) @ K.T
        return z @ K.T


class ToyTemporal(nn.Module):
    def __init__(self, affine=False):
        super().__init__()
        self.backbone = ToyBackbone()
        self.koop = ToyKoop(affine=affine)

    def forward(self, x, return_states=False):
        z = self.backbone(x)
        states = []
        for _ in range(3):
            states.append(z)
            z = self.koop.step(z)
        states = torch.stack(states, dim=1)
        pred = states.transpose(1, 2).reshape(x.shape[0], -1)
        return (pred, states) if return_states else pred


class TestExplainability(unittest.TestCase):
    def setUp(self):
        self.model = LinearForecast().eval()
        self.x = torch.tensor([[[1., 2.], [2., 1.], [3., 4.]]])
        self.base = torch.zeros(1, 1, 2)

    def test_target_feature_major(self):
        y = self.model(self.x)
        actual = target_prediction(y, channels=2, horizon=2, target_channel=0, target_step=1)
        self.assertEqual(actual.shape, (1,))
        self.assertAlmostEqual(actual.item(), (self.x * self.model.weights).sum().item())

    def test_ig_linear_exact(self):
        result = integrated_gradients(self.model, self.x, self.base, horizon=2,
                                      target_channel=0, target_step=1, steps=12)
        torch.testing.assert_close(result.attribution, self.x * self.model.weights)
        self.assertLess(abs(result.completeness_error.item()), 1e-5)

    def test_ig_baseline_equal_input(self):
        result = integrated_gradients(self.model, self.x, self.x, horizon=2,
                                      target_channel=0, target_step=1, steps=5)
        torch.testing.assert_close(result.attribution, torch.zeros_like(self.x))
        self.assertAlmostEqual(result.completeness_error.item(), 0.0)

    def test_ig_reject_train_mode(self):
        self.model.train()
        with self.assertRaises(ValueError):
            integrated_gradients(self.model, self.x, self.base, horizon=2, steps=8)

    def test_occlusion_linear_block(self):
        result = temporal_occlusion(self.model, self.x, self.base, horizon=2,
                                    target_channel=0, target_step=1, block_size=1)
        torch.testing.assert_close(result.impacts, self.x * self.model.weights)

    def test_occlusion_multiple_steps(self):
        result = temporal_occlusion(self.model, self.x, self.base, horizon=2,
                                    target_channel=0, target_step=1, block_size=2)
        expected = torch.zeros_like(self.x)
        for c in range(2):
            expected[:, 0:2, c] = (self.x[:, 0:2, c] * self.model.weights[None, 0:2, c]).sum(dim=1)[:, None]
            expected[:, 2:3, c] = self.x[:, 2:3, c] * self.model.weights[None, 2:3, c]
        torch.testing.assert_close(result.impacts, expected)

    def test_koopman_affine(self):
        model = ToyTemporal(affine=True).eval()
        x = torch.ones(2, 5, 2)
        y = torch.ones(2, 6)
        diagnostics = koopman_diagnostics(model, x, y, horizon=3, variant="M4")
        self.assertEqual(tuple(diagnostics.states.shape), (2, 3, 2))
        self.assertEqual(tuple(diagnostics.shift_gap.shape), (2, 3))
        self.assertTrue(torch.isnan(diagnostics.shift_gap[:, 0]).all())
        self.assertAlmostEqual(diagnostics.spectral_radius, .7, places=5)
        self.assertIsNotNone(diagnostics.equilibrium)
        self.assertTrue(torch.all(diagnostics.centered_norm[:, 1] < diagnostics.centered_norm[:, 0]))

    def test_koopman_m0_skipped(self):
        self.assertIsNone(koopman_diagnostics(self.model, self.x, torch.zeros(1, 4),
                                               horizon=2, variant="M0"))

    def test_sample_selection_reproducible(self):
        self.assertEqual(select_indices(101, 5).tolist(), [0, 25, 50, 75, 100])
        self.assertEqual(select_indices(3, 8).tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
