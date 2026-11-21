import numpy as np

from model.evaluate import compare
from model.tests.conftest import pattern_batch


def test_compare_reports_losses_and_agreement(trained_small):
    model, im, _ = trained_small
    seq = pattern_batch(np.random.default_rng(3), 4, 32, 24)
    xs, ys = seq[:, :-1], seq[:, 1:]
    rep = compare(model, im, xs, ys)
    assert set(rep) >= {"qat_loss", "intref_loss", "rel_diff", "top1_agree", "max_logit_err",
                        "mean_logit_err", "n_targets"}
    assert rep["n_targets"] == ys.size
    assert abs(rep["rel_diff"]) < 0.05
    assert rep["top1_agree"] > 0.95
