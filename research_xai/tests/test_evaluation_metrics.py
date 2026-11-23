"""Kiểm thử các công thức trong `research_xai/evaluation/metrics.py`.

Chạy: python -m unittest research_xai.tests.test_evaluation_metrics -v
"""
import unittest

import torch
from torch import nn

from research_xai.evaluation.metrics import (
    RegressionAccumulator,
    compare_pair,
    evaluate_model,
    relative_improvement_pct,
    summarize_early_stopping,
    summarize_model,
    summarize_pairs,
)


class IdentityForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class TestEvaluationMetrics(unittest.TestCase):
    def test_mse_mae_elementwise(self):
        pred = torch.tensor([[1., 3.], [4., 6.]])
        true = torch.tensor([[1., 2.], [2., 4.]])
        acc = RegressionAccumulator()
        acc.update(pred[:1], true[:1])
        acc.update(pred[1:], true[1:])
        result = acc.compute()
        # Errors [0, 1, 2, 2]: MSE=9/4; MAE=5/4.
        self.assertAlmostEqual(result["test_mse"], 2.25)
        self.assertAlmostEqual(result["test_mae"], 1.25)

    def test_evaluate_model_unequal_minibatches(self):
        model = IdentityForecaster()
        x = torch.tensor([[[1., 2.]], [[2., 4.]], [[3., 6.]]])
        y = torch.tensor([[1., 2.], [2., 6.], [0., 6.]])
        # batch_size=2 -> minibatch cuối có 1 mẫu
        scores = evaluate_model(model, x, y, batch_size=2)
        self.assertAlmostEqual(scores["test_mse"], (4+9)/6)
        self.assertAlmostEqual(scores["test_mae"], (2+3)/6)

    def test_input_shape_mismatch(self):
        acc=RegressionAccumulator()
        with self.assertRaises(ValueError):
            acc.update(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_paired_relative_improvement(self):
        b={"dataset":"Electricity","lookback":100,"horizon":8,"seed":7,"model":"M1","test_mse":0.04}
        c={**b,"model":"M4","test_mse":0.03}
        s=compare_pair(b,c,"M1","M4")
        self.assertEqual(s["candidate_wins"],1)
        self.assertAlmostEqual(s["relative_improvement_pct"],25.0)
        summary=summarize_pairs([s])
        self.assertEqual(summary["candidate_wins"],1)
        self.assertAlmostEqual(summary["improvement_of_mean_mse_pct"],25.0)
        c["horizon"]=16
        with self.assertRaises(ValueError):
            compare_pair(b,c,"M1","M4")

    def test_early_stopping(self):
        rec={"best_epoch":100,"stopped_epoch":900,"epochs_saved":3100,"early_stopped":True}
        stat=summarize_early_stopping([rec],target_epochs=4000)
        self.assertAlmostEqual(stat["epoch_budget_saved_pct"],77.5)
        self.assertEqual(stat["early_stopped_count"],1)

    def test_mean_and_median(self):
        r1={"test_mse":0.1,"test_mae":0.2,"best_val_mse":0.03}
        r2={"test_mse":0.3,"test_mae":0.4,"best_val_mse":0.05}
        r=summarize_model([r1,r2])
        self.assertAlmostEqual(r["mean_test_mse"],0.2)
        self.assertAlmostEqual(r["mean_test_mae"],0.3)


if __name__ == "__main__":
    unittest.main()
