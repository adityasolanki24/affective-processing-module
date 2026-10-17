"""Learning stack: masked features, observed-outcome labels, metrics, baselines, shadow promotion and rollback."""
import numpy as np
import pytest
from conftest import Robot

from apm.learning import baselines
from apm.learning.dataset import Episode, armed_mask, episode_from_stream, samples
from apm.learning.evaluation import auroc, bootstrap_difference, brier, ece, recall_at_fpr
from apm.learning.features import NAMES, WIDTH, FeatureTracker
from apm.learning.replay import ReplayBuffer, ShadowLearner

F = len(NAMES)


def test_missing_inputs_are_masked_not_zero_filled():
    robot = Robot()
    tracker = FeatureTracker(robot.profile)
    tracker.ingest(robot.self_state(1.0, stability=None, energy_v=12.6))
    v = tracker.vector(1.0)
    assert v.shape == (WIDTH,)
    assert v[F + 0] == 0 and v[0] == 0          # stability missing → mask 0
    assert v[F + 4] == 1 and v[4] == 1.0        # energy calibrated: nominal → 1
    tracker.ingest(robot.entity(1.0, distance=1.0, closing=0.5))
    v = tracker.vector(1.0)
    assert v[F + 5] == 1 and abs(v[5] - np.exp(-1)) < 1e-9 and abs(v[6] - 0.25) < 1e-9


def test_labels_come_only_from_observed_outcomes_within_the_horizon():
    robot = Robot()

    def stream():
        yield 0.0, [robot.profile]
        for i in range(100):
            t = 0.1 * (i + 1)
            msgs = [robot.self_state(t)]
            if abs(t - 3.0) < 1e-9:
                msgs.append(robot.hypothesis(t, kind="stability_loss", target=None, p=0.99))  # prediction: no label
            if abs(t - 6.0) < 1e-9:
                msgs.append(robot.outcome(t, kind="stability_loss", target=None, oid="loss"))
            yield t, msgs

    e = episode_from_stream("x", robot.profile, stream(), 0.1, meta={})
    X, y, g, ts = samples([e], "stability_loss", 0.5, history=5)
    positive_times = ts[y == 1]
    assert np.allclose(positive_times, [5.5, 5.6, 5.7, 5.8, 5.9], atol=1e-6)  # (t, t + 0.5] contains 6.0
    assert not np.any((ts >= 6.0) & (ts <= 7.0))  # not armed during the loss (onset + 1 s re-arm)
    assert X.shape == (len(y), 5, WIDTH)


def test_armed_mask_excludes_after_fall():
    times = np.arange(0, 10, 0.5)
    mask = armed_mask(times, {"stability_loss": [3.0], "recovery": [4.0], "fall": [7.0]})
    assert mask[times < 3].all() and not mask[(times >= 3) & (times <= 4)].any() and not mask[times >= 7].any()


def test_metrics_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert brier([0, 0, 1, 1], y) == 0 and brier([.5] * 4, y) == .25
    assert auroc([.1, .2, .8, .9], y) == 1.0 and auroc([.9, .8, .2, .1], y) == 0.0 and auroc([.5] * 4, y) == .5
    assert ece([0, 0, 1, 1], y) == 0 and abs(ece([.5] * 4, y)) < 1e-12
    assert recall_at_fpr(np.r_[np.linspace(0, .5, 100), [.9, .95]], np.r_[np.zeros(100), [1, 1]]) == 1.0
    d = bootstrap_difference(np.array([.1, .9, .1, .9]), np.array([.5] * 4), np.array([0, 1, 0, 1]),
                             np.array([0, 0, 1, 1]), n=50)
    assert d["difference"] < 0 and d["ci95"][1] < 0


def test_statistical_baselines_learn_a_margin_dependence():
    rng = np.random.default_rng(0)
    n = 4000
    X = np.zeros((n, 1, WIDTH), dtype=np.float32)
    margin = rng.uniform(0, 1, n)
    X[:, 0, 0], X[:, 0, F] = margin, 1
    y = (rng.uniform(size=n) < np.where(margin < .3, .6, .02)).astype(float)
    for model in (baselines.MarginBins(), baselines.Logistic()):
        p = model.fit(X, y).predict(X)
        assert p[margin < .2].mean() > 3 * p[margin > .5].mean()
    assert baselines.parameter_count(baselines.Logistic().fit(X, y)) == WIDTH + 1


class _Constant:
    """Fake model: predicts a learned constant (enough to exercise promotion logic)."""

    def __init__(self, value=.5):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)

    def clone(self):
        return _Constant(self.value)

    def fit(self, X, y, epochs=None, early_stopping=False):
        self.value = float(np.mean(y))
        return self


def test_shadow_learner_promotes_rejects_and_rolls_back():
    rng = np.random.default_rng(0)
    y_val = (rng.uniform(size=400) < .1).astype(float)
    X_val = np.zeros((400, 1, 1))
    learner = ShadowLearner(validated=_Constant(.5), X_val=X_val, y_val=y_val, protected=np.arange(400) < 200)
    good = learner.offer(np.zeros((300, 1, 1)), (rng.uniform(size=300) < .1).astype(float), 1)
    assert good["promoted"] and abs(learner.validated.value - .1) < .05
    served = learner.validated
    bad = learner.offer(np.zeros((300, 1, 1)), np.ones(300), 2)  # would predict 1.0 everywhere
    assert not bad["promoted"] and learner.validated is served  # live model untouched
    assert learner.rollback() and learner.validated.value == .5
    buffer = ReplayBuffer()
    buffer.add(np.zeros((10, 1, 1)), np.zeros(10), 0)
    Xm, ym = buffer.mix(np.ones((10, 1, 1)), np.ones(10), .5, rng)
    assert len(ym) == 20 and ym.sum() == 10


@pytest.mark.learning
def test_gru_learns_a_temporal_precursor():
    torch = pytest.importorskip("torch")
    from apm.learning.predictor import GRUPredictor
    rng = np.random.default_rng(0)
    n, T = 3000, 10
    X = rng.normal(0, .1, (n, T, 2)).astype(np.float32)
    y = (rng.uniform(size=n) < .5).astype(np.float32)
    X[y == 1, 3, 1] += 1.0  # the label is only visible in the history, not at the last step
    model = GRUPredictor(2, hidden=16, head=8, epochs=40, batch=64, seed=0).fit(X[:2000], y[:2000], X[2000:], y[2000:])
    assert brier(model.predict(X[2000:]), y[2000:]) < .05
    assert model.parameters < 2000 and torch.__version__
    assert episode_types_ok()


def episode_types_ok():
    return Episode.__dataclass_fields__.keys() >= {"times", "features", "events"}


def test_auroc_vectorised_ties_equal_reference():
    import numpy as np

    from apm.learning.evaluation import auroc

    def reference(p, y):
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
        pos, neg = p[y], p[~y]
        order = np.argsort(np.concatenate([pos, neg]))
        ranks = np.empty(len(order))
        ranks[order] = np.arange(1, len(order) + 1)
        values = np.concatenate([pos, neg])
        for v in np.unique(values):
            sel = values == v
            ranks[sel] = ranks[sel].mean()
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.round(rng.random(300), int(rng.integers(1, 4)))  # many ties
        y = rng.random(300) < 0.3
        assert auroc(p, y) == reference(p, y)
