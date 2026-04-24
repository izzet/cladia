import numpy as np

from dfdiagnoser_ml.metrics import (
	compute_asymmetric_miss_score,
	compute_interval_quality_score,
	compute_winkler_score,
	compute_prediction_interval_coverage_probability,
	lenient_overlap_accuracy,
)


def _synth_intervals(n: int = 200, seed: int = 0, skew: bool = False):
	"""Synthesize true and base predicted intervals with controllable miss/waste and optional skew.

	Returns
	-------
	(y_true, y_pred_base, y_point, y_obs)
	- y_true: (N,2) true intervals
	- y_pred_base: (N,2) base predicted intervals with bias/noise
	- y_point: (N,) scalar points (true mean; for skew=True, shift mean upward)
	- y_obs: (N,M) synthetic observations per sample for PICP (M=10)
	"""
	rng = np.random.default_rng(seed)
	centers = rng.uniform(100.0, 1000.0, size=n)
	true_width = rng.uniform(40.0, 160.0, size=n)
	y_true = np.column_stack([centers - true_width/2.0, centers + true_width/2.0])
	# Base predicted with multiplicative width noise and additive center bias
	width_bias = rng.normal(0.0, 0.15, size=n)  # ~15% width distortion
	center_bias = rng.normal(0.0, 0.10, size=n)  # ~10% of width shift (normalized later)
	pred_width = true_width * np.clip(1.0 + width_bias, 0.5, 2.0)
	pred_center = centers + center_bias * true_width
	y_pred = np.column_stack([pred_center - pred_width/2.0, pred_center + pred_width/2.0])
	# y_point (mean) optionally skewed upward so mean exits the IQR for a subset
	y_point = centers.copy()
	if skew:
		# Push mean beyond the upper quartile for 30% of samples
		mask = rng.random(n) < 0.3
		y_point[mask] = y_point[mask] + 0.35 * true_width[mask]
	# Synthetic observations: 10 points per sample, 5 inside true interval, 5 outside (controlled)
	M = 10
	y_obs = np.zeros((n, M), dtype=float)
	for i in range(n):
		amin, amax = y_true[i]
		mid = (amin + amax) / 2.0
		# 5 inside uniformly within [amin, amax]
		inside = rng.uniform(amin, amax, size=M//2)
		# 5 outside split: below and above with small margin
		below = amin - rng.uniform(0.05, 0.25) * true_width[i]
		above = amax + rng.uniform(0.05, 0.25) * true_width[i]
		outside = np.array([below, below * 0 + (amin - 0.1*true_width[i]), above, above * 0 + (amax + 0.1*true_width[i]), mid + 2.0*true_width[i]])
		vals = np.concatenate([inside, outside])
		rng.shuffle(vals)
		y_obs[i, :] = vals
	return y_true, y_pred, y_point, y_obs


def _apply_calibration(y_pred: np.ndarray, scale: float, shift_frac: float, y_true: np.ndarray) -> np.ndarray:
	"""Rescale width by 'scale' and shift center by 'shift_frac' * mean(true_width)."""
	pred_center = (y_pred[:, 0] + y_pred[:, 1]) / 2.0
	pred_width = (y_pred[:, 1] - y_pred[:, 0]) * max(scale, 0.0)
	true_w = y_true[:, 1] - y_true[:, 0]
	shift = shift_frac * float(np.mean(true_w))
	L = pred_center - pred_width/2.0 + shift
	U = pred_center + pred_width/2.0 + shift
	L2 = np.minimum(L, U)
	U2 = np.maximum(L, U)
	return np.column_stack([L2, U2])


def _tune_calibration(y_true: np.ndarray, y_pred: np.ndarray, metric: str, y_point: np.ndarray, y_obs: np.ndarray, alpha: float = 1.0):
	"""Grid search for (scale, shift_frac) to minimize a given metric on a validation split.

	metric in { 'AMS', 'WINKLER', 'IQS' }
	"""
	n = y_true.shape[0]
	idx = np.arange(n)
	val_mask = (idx % 2) == 0
	# Small grids keep tests fast and stable
	scales = [0.8, 1.0, 1.2]
	shifts = [-0.1, 0.0, 0.1]
	best = (None, np.inf)
	for s in scales:
		for d in shifts:
			P = _apply_calibration(y_pred, s, d, y_true)
			if metric == 'AMS':
				val = compute_asymmetric_miss_score(y_true[val_mask], P[val_mask], alpha=alpha)
			elif metric == 'WINKLER':
				val = compute_winkler_score(P[val_mask], y_point[val_mask], alpha=0.5, include_endpoints=True)
			elif metric == 'IQS':
				val = compute_interval_quality_score(y_true[val_mask], P[val_mask], lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
			else:
				raise ValueError('Unknown metric')
			if val < best[1]:
				best = ((s, d), val)
	return best[0]


def test_ams_calibration_yields_lowest_ams_and_good_tradeoffs():
	# Synthetic with modest skew and interval distortions
	y_true, y_pred, y_point, y_obs = _synth_intervals(n=160, seed=7, skew=True)
	# Calibrate each objective on validation split
	cal_ams = _tune_calibration(y_true, y_pred, 'AMS', y_point, y_obs, alpha=2.0)
	cal_wink = _tune_calibration(y_true, y_pred, 'WINKLER', y_point, y_obs, alpha=2.0)
	cal_iqs = _tune_calibration(y_true, y_pred, 'IQS', y_point, y_obs, alpha=2.0)
	# Apply on held-out (odd indices)
	odd = (np.arange(len(y_true)) % 2) == 1
	P_ams = _apply_calibration(y_pred, cal_ams[0], cal_ams[1], y_true)
	P_wink = _apply_calibration(y_pred, cal_wink[0], cal_wink[1], y_true)
	P_iqs = _apply_calibration(y_pred, cal_iqs[0], cal_iqs[1], y_true)
	# Metrics on test split
	ams_ams = compute_asymmetric_miss_score(y_true[odd], P_ams[odd], alpha=2.0)
	ams_wink = compute_asymmetric_miss_score(y_true[odd], P_wink[odd], alpha=2.0)
	ams_iqs = compute_asymmetric_miss_score(y_true[odd], P_iqs[odd], alpha=2.0)
	# AMS-tuned should be best on AMS (by construction, but on held-out)
	assert ams_ams <= ams_wink + 1e-9
	assert ams_ams <= ams_iqs + 1e-9
	# Tradeoffs: AMS should maintain strong overlap while controlling width
	loa_ams = lenient_overlap_accuracy(y_true[odd], P_ams[odd])
	loa_wink = lenient_overlap_accuracy(y_true[odd], P_wink[odd])
	# Not necessarily bigger, but should not collapse
	assert loa_ams >= 0.6


def test_ams_reduces_undercoverage_with_alpha_gt_1():
	y_true, y_pred, y_point, y_obs = _synth_intervals(n=120, seed=9, skew=True)
	cal_a1 = _tune_calibration(y_true, y_pred, 'AMS', y_point, y_obs, alpha=1.0)
	cal_a2 = _tune_calibration(y_true, y_pred, 'AMS', y_point, y_obs, alpha=2.0)
	odd = (np.arange(len(y_true)) % 2) == 1
	P_a1 = _apply_calibration(y_pred, cal_a1[0], cal_a1[1], y_true)
	P_a2 = _apply_calibration(y_pred, cal_a2[0], cal_a2[1], y_true)
	# Estimate under- and over-coverage using PICP components relative to y_obs
	# Approximate under-coverage: fraction of observations below L; over-coverage: fraction above U
	L1, U1 = P_a1[odd, 0], P_a1[odd, 1]
	L2, U2 = P_a2[odd, 0], P_a2[odd, 1]
	Y = y_obs[odd]
	under1 = float(np.mean(Y < L1[:, None]))
	under2 = float(np.mean(Y < L2[:, None]))
	# With alpha=2, we expect reduced under-coverage relative to alpha=1
	assert under2 <= under1 + 1e-9


def test_ams_calibration_keeps_picp_reasonable_and_controls_width():
	y_true, y_pred, y_point, y_obs = _synth_intervals(n=100, seed=11, skew=False)
	cal_ams = _tune_calibration(y_true, y_pred, 'AMS', y_point, y_obs, alpha=1.5)
	odd = (np.arange(len(y_true)) % 2) == 1
	P_ams = _apply_calibration(y_pred, cal_ams[0], cal_ams[1], y_true)
	# PICP near 0.5 for IQR-like intervals
	picp = compute_prediction_interval_coverage_probability(P_ams[odd], y_obs[odd], include_endpoints=True)
	assert 0.35 <= picp <= 0.65
	# Control width: mean predicted width should not explode relative to true width
	true_w = (y_true[odd, 1] - y_true[odd, 0]).mean()
	pred_w = (P_ams[odd, 1] - P_ams[odd, 0]).mean()
	assert pred_w / true_w < 1.6





