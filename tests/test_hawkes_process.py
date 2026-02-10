import numpy as np
import pandas as pd
import pytest

from hawkes_process import (
    detect_clusters,
    extract_events,
    fit_hawkes,
    hawkes_intensity,
    hawkes_var_adjustment,
    simulate_hawkes,
)


@pytest.fixture(scope="module")
def clustered_returns():
    """Returns with injected clusters at indices 50-54 and 100-102."""
    rng = np.random.default_rng(42)
    r = rng.normal(0, 1.5, 300)
    r[50:55] = rng.normal(-5, 1, 5)
    r[100:103] = rng.normal(-4, 0.5, 3)
    return pd.Series(r, index=pd.date_range("2024-01-01", periods=300, freq="D"))


@pytest.fixture(scope="module")
def hawkes_events(clustered_returns):
    return extract_events(clustered_returns, threshold_percentile=5.0, use_absolute=True)


@pytest.fixture(scope="module")
def hawkes_params(hawkes_events):
    return fit_hawkes(hawkes_events["event_times"], hawkes_events["T"])


# ── extract_events ────────────────────────────────────────────────────


class TestExtractEvents:
    def test_output_keys(self, hawkes_events):
        expected = {"event_times", "event_returns", "event_indices", "threshold", "n_events", "T", "dates"}
        assert expected == set(hawkes_events.keys())

    def test_events_detected(self, hawkes_events):
        assert hawkes_events["n_events"] >= 5

    def test_threshold_positive(self, hawkes_events):
        assert hawkes_events["threshold"] > 0

    def test_left_tail_only(self, clustered_returns):
        ev = extract_events(clustered_returns, threshold_percentile=5.0, use_absolute=False)
        assert all(r < 0 for r in ev["event_returns"])

    def test_T_equals_length(self, clustered_returns, hawkes_events):
        assert hawkes_events["T"] == float(len(clustered_returns))


# ── fit_hawkes ────────────────────────────────────────────────────────


class TestFitHawkes:
    def test_params_positive(self, hawkes_params):
        assert hawkes_params["mu"] > 0
        assert hawkes_params["alpha"] > 0
        assert hawkes_params["beta"] > 0

    def test_stationary(self, hawkes_params):
        assert hawkes_params["stationary"] is True
        assert hawkes_params["branching_ratio"] < 1.0

    def test_half_life_positive(self, hawkes_params):
        assert hawkes_params["half_life"] > 0

    def test_aic_bic_finite(self, hawkes_params):
        assert np.isfinite(hawkes_params["aic"])
        assert np.isfinite(hawkes_params["bic"])

    def test_too_few_events_raises(self):
        with pytest.raises(ValueError, match="Need ≥5"):
            fit_hawkes(np.array([1.0, 2.0, 3.0]), T=100.0)

    def test_output_keys(self, hawkes_params):
        expected = {"mu", "alpha", "beta", "branching_ratio", "log_likelihood",
                    "aic", "bic", "n_events", "T", "half_life", "stationary"}
        assert expected == set(hawkes_params.keys())


# ── hawkes_intensity ──────────────────────────────────────────────────


class TestHawkesIntensity:
    def test_intensity_ge_baseline(self, hawkes_events, hawkes_params):
        result = hawkes_intensity(hawkes_events["event_times"], hawkes_params)
        assert result["current_intensity"] >= hawkes_params["mu"] - 1e-10

    def test_peak_above_baseline(self, hawkes_events, hawkes_params):
        result = hawkes_intensity(hawkes_events["event_times"], hawkes_params)
        assert result["peak_intensity"] > hawkes_params["mu"]

    def test_output_keys(self, hawkes_events, hawkes_params):
        result = hawkes_intensity(hawkes_events["event_times"], hawkes_params)
        expected = {"t_eval", "intensity", "current_intensity", "baseline",
                    "intensity_ratio", "peak_intensity", "mean_intensity"}
        assert expected == set(result.keys())


# ── hawkes_var_adjustment ─────────────────────────────────────────────


class TestHawkesVarAdjustment:
    def test_multiplier_ge_one_when_elevated(self):
        adj = hawkes_var_adjustment(-3.0, current_intensity=0.1, baseline_intensity=0.05)
        assert adj["multiplier"] >= 1.0
        assert adj["adjusted_var"] <= adj["base_var"]  # more negative

    def test_no_adjustment_at_baseline(self):
        adj = hawkes_var_adjustment(-3.0, current_intensity=0.05, baseline_intensity=0.05)
        assert abs(adj["multiplier"] - 1.0) < 1e-10
        assert abs(adj["adjusted_var"] - adj["base_var"]) < 1e-10

    def test_capped(self):
        adj = hawkes_var_adjustment(-2.0, current_intensity=1.0, baseline_intensity=0.1, max_multiplier=3.0)
        assert adj["capped"] is True
        assert abs(adj["multiplier"] - 3.0) < 1e-10

    def test_base_var_preserved(self):
        adj = hawkes_var_adjustment(-5.5, current_intensity=0.05, baseline_intensity=0.05)
        assert adj["base_var"] == -5.5


# ── detect_clusters ───────────────────────────────────────────────────


class TestDetectClusters:
    def test_clusters_found(self, hawkes_events, hawkes_params):
        clusters = detect_clusters(hawkes_events["event_times"], hawkes_params)
        assert len(clusters) >= 1

    def test_cluster_fields(self, hawkes_events, hawkes_params):
        clusters = detect_clusters(hawkes_events["event_times"], hawkes_params)
        if clusters:
            c = clusters[0]
            assert "cluster_id" in c
            assert "n_events" in c and c["n_events"] >= 2
            assert c["duration"] >= 0

    def test_no_clusters_in_uniform(self, hawkes_params):
        # Events spaced far apart — no clusters
        events = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])
        clusters = detect_clusters(events, hawkes_params, gap_threshold=2.0)
        assert len(clusters) == 0


# ── simulate_hawkes ───────────────────────────────────────────────────


class TestSimulateHawkes:
    def test_produces_events(self, hawkes_params):
        sim = simulate_hawkes(hawkes_params, T=500.0, seed=99)
        assert sim["n_events"] >= 0
        assert sim["T"] == 500.0

    def test_deterministic_with_seed(self, hawkes_params):
        s1 = simulate_hawkes(hawkes_params, T=300.0, seed=42)
        s2 = simulate_hawkes(hawkes_params, T=300.0, seed=42)
        assert s1["n_events"] == s2["n_events"]
        np.testing.assert_array_equal(s1["event_times"], s2["event_times"])

    def test_intensity_path_length(self, hawkes_params):
        sim = simulate_hawkes(hawkes_params, T=100.0, seed=7)
        assert len(sim["intensity_path"]) == len(sim["intensity_t"])
        assert len(sim["intensity_path"]) >= 100

