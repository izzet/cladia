import numpy as np
from typing import List
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression

# ------------------------------
# Custom metrics/scorers
# ------------------------------

def strict_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_pred), 1e-8)))


def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.median(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))))


def lenient_overlap_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    overlaps: List[bool] = []
    for i in range(len(y_true)):
        a_min, a_max = y_true[i]
        p_min, p_max = y_pred[i]
        intersection = max(0, min(a_max, p_max) - max(a_min, p_min))
        overlaps.append(intersection > 0)
    return float(np.mean(overlaps))


def lenient_overlap_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return lenient_overlap_accuracy(y_true, y_pred)


# ------------------------------
# Interval helpers and extended metrics (IQS, Winkler, MIR, PICP)
# ------------------------------

def _interval_intersection_length(a: np.ndarray, b: np.ndarray) -> float:
	"""Return the length of intersection between intervals a=[amin,amax] and b=[bmin,bmax]."""
	amin, amax = float(a[0]), float(a[1])
	bmin, bmax = float(b[0]), float(b[1])
	return max(0.0, min(amax, bmax) - max(amin, bmin))


def lenient_overlap_accuracy_tolerant(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	tolerance: float = 0.0,
	mode: str = 'relative_to_true_width',
	include_endpoints: bool = False,
) -> float:
	"""Compute overlap accuracy under tolerance.

	Parameters
	----------
	y_true : np.ndarray of shape (N, 2)
		True intervals [a_min, a_max].
	y_pred : np.ndarray of shape (N, 2)
		Predicted intervals [p_min, p_max].
	tolerance : float
		Tolerance amount. If mode=='relative_to_true_width', expands true interval by
		± (tolerance * true_width). If mode=='scale_endpoints', expands by scaling endpoints
		[a_min*(1-tol), a_max*(1+tol)]. Default 0.0 (strict).
	mode : {'relative_to_true_width', 'scale_endpoints'}
		How to apply tolerance to the true interval.
	include_endpoints : bool
		If True, zero-length touching intervals count as overlap. Default False (strict > 0).

	Returns
	-------
	float
		Mean of per-sample overlap indicators under tolerance.
	"""
	assert y_true.shape == y_pred.shape and y_true.shape[1] == 2, "Intervals must be (N,2)"
	assert tolerance >= 0.0, "tolerance must be non-negative"
	N = y_true.shape[0]
	matches: list = []
	for i in range(N):
		amin, amax = float(y_true[i, 0]), float(y_true[i, 1])
		pmin, pmax = float(y_pred[i, 0]), float(y_pred[i, 1])
		if mode == 'relative_to_true_width':
			w = max(0.0, amax - amin)
			rmin, rmax = amin - tolerance*w, amax + tolerance*w
		elif mode == 'scale_endpoints':
			rmin, rmax = amin*(1.0 - tolerance), amax*(1.0 + tolerance)
		else:
			raise ValueError("mode must be 'relative_to_true_width' or 'scale_endpoints'")
		inter = min(pmax, rmax) - max(pmin, rmin)
		matches.append(inter >= 0.0 if include_endpoints else inter > 0.0)
	return float(np.mean(matches))


def compute_mean_in_interval_rate(
	y_pred: np.ndarray,
	y_point: np.ndarray,
	include_endpoints: bool = True,
) -> float:
	"""Mean-in-interval rate (MIR): fraction of scalar points inside predicted intervals.

	Parameters
	----------
	y_pred : np.ndarray of shape (N, 2)
		Predicted intervals [L, U].
	y_point : np.ndarray of shape (N,)
		Scalar point per sample (e.g., true mean per item).
	include_endpoints : bool
		If True, count L and U as inside.
	"""
	assert y_pred.ndim == 2 and y_pred.shape[1] == 2, "y_pred must be (N,2)"
	y_point = np.asarray(y_point).reshape(-1)
	assert y_point.shape[0] == y_pred.shape[0], "y_point length must match N"
	L = y_pred[:, 0]
	U = y_pred[:, 1]
	if include_endpoints:
		inside = (y_point >= L) & (y_point <= U)
	else:
		inside = (y_point > L) & (y_point < U)
	return float(np.mean(inside))


def compute_winkler_score(
	y_pred: np.ndarray,
	y_point: np.ndarray,
	alpha: float = 0.5,
	include_endpoints: bool = True,
) -> float:
	"""Compute mean Winkler interval score S_alpha for scalar observations.

	S_alpha(L,U;y) = (U-L) if L <= y <= U,
	otherwise (U-L) + (2/alpha) * distance outside the interval.

	Parameters
	----------
	y_pred : np.ndarray (N,2)
	y_point : np.ndarray (N,)
	alpha : float
		Nominal level (e.g., 0.5 for IQR). Must be in (0,1].
	include_endpoints : bool
		Whether L and U count as inside.
	"""
	assert 0.0 < alpha <= 1.0, "alpha must be in (0,1]"
	assert y_pred.ndim == 2 and y_pred.shape[1] == 2, "y_pred must be (N,2)"
	y_point = np.asarray(y_point).reshape(-1)
	assert y_point.shape[0] == y_pred.shape[0], "y_point length must match N"
	L = y_pred[:, 0]
	U = y_pred[:, 1]
	w = U - L
	if include_endpoints:
		inside = (y_point >= L) & (y_point <= U)
	else:
		inside = (y_point > L) & (y_point < U)
	outside_below = y_point < L
	outside_above = y_point > U
	pen = np.zeros_like(y_point, dtype=float)
	pen[outside_below] = L[outside_below] - y_point[outside_below]
	pen[outside_above] = y_point[outside_above] - U[outside_above]
	scores = np.where(inside, w, w + (2.0/alpha) * pen)
	return float(np.mean(scores))


def compute_interval_quality_score(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	lambda_weight: float = 1.0,
	tolerance: float = 0.0,
	width_normalization: str = 'dataset_mean',
	tolerance_mode: str = 'relative_to_true_width',
) -> float:
	"""Interval Quality Score (IQS): (1 - LOA) + lambda * MAE_width_norm.

	LOA is computed as overlap accuracy; if tolerance > 0, a tolerant LOA is used.

	width_normalization options:
	- 'dataset_mean': divide absolute width error by mean(true_width) across dataset
	- 'by_true_width': divide by each sample's true width before averaging
	"""
	assert y_true.shape == y_pred.shape and y_true.shape[1] == 2, "Intervals must be (N,2)"
	assert lambda_weight >= 0.0, "lambda_weight must be non-negative"
	true_w = y_true[:, 1] - y_true[:, 0]
	pred_w = y_pred[:, 1] - y_pred[:, 0]
	width_err = np.abs(pred_w - true_w)
	if width_normalization == 'dataset_mean':
		denom = float(np.mean(np.maximum(true_w, 1e-12)))
		mae_width_norm = float(np.mean(width_err)) / denom if denom > 0 else float(np.mean(width_err))
	elif width_normalization == 'by_true_width':
		mae_width_norm = float(np.mean(width_err / np.maximum(true_w, 1e-12)))
	else:
		raise ValueError("width_normalization must be 'dataset_mean' or 'by_true_width'")
	if tolerance > 0.0:
		loa = lenient_overlap_accuracy_tolerant(y_true, y_pred, tolerance=tolerance, mode=tolerance_mode, include_endpoints=False)
	else:
		loa = lenient_overlap_accuracy(y_true, y_pred)
	return (1.0 - float(loa)) + float(lambda_weight) * float(mae_width_norm)


def compute_prediction_interval_coverage_probability(
	y_pred: np.ndarray,
	y_obs: np.ndarray,
	include_endpoints: bool = True,
) -> float:
	"""Compute PICP: fraction of observations falling inside predicted intervals.

	Supports two shapes:
	- y_pred shape (1,2), y_obs shape (M,) => coverage over M observations for one interval
	- y_pred shape (N,2), y_obs shape (N,M) => mean coverage across N intervals, each with M obs
	"""
	y_pred = np.asarray(y_pred)
	y_obs = np.asarray(y_obs)
	assert y_pred.ndim == 2 and y_pred.shape[1] == 2, "y_pred must be (N,2)"
	L = y_pred[:, 0]
	U = y_pred[:, 1]
	if y_obs.ndim == 1:
		# Two supported cases for 1D y_obs:
		# (a) Global coverage over M observations for a single interval: y_pred shape (1,2)
		# (b) Per-sample scalar observation (e.g., mean) for each interval: y_pred shape (N,2) and y_obs shape (N,)
		if y_pred.shape[0] == 1:
			if include_endpoints:
				inside = (y_obs >= L[0]) & (y_obs <= U[0])
			else:
				inside = (y_obs > L[0]) & (y_obs < U[0])
			return float(np.mean(inside))
		elif y_obs.shape[0] == y_pred.shape[0]:
			if include_endpoints:
				inside = (y_obs >= L) & (y_obs <= U)
			else:
				inside = (y_obs > L) & (y_obs < U)
			return float(np.mean(inside))
		else:
			raise AssertionError("For 1D y_obs, expected y_pred shape (1,2) or matching length (N,) for per-sample coverage")
	elif y_obs.ndim == 2:
		assert y_obs.shape[0] == y_pred.shape[0], "y_obs must have same first dim N as y_pred"
		if include_endpoints:
			inside = (y_obs >= L[:, None]) & (y_obs <= U[:, None])
		else:
			inside = (y_obs > L[:, None]) & (y_obs < U[:, None])
		per_sample_cov = np.mean(inside, axis=1)
		return float(np.mean(per_sample_cov))
	else:
		raise ValueError("y_obs must be 1D or 2D array")


def compute_mean_prediction_interval_width(y_pred: np.ndarray) -> float:
	"""MPIW: mean prediction interval width over samples.

	Parameters
	----------
	y_pred : np.ndarray (N,2)
		Predicted intervals.
	"""
	y_pred = np.asarray(y_pred)
	assert y_pred.ndim == 2 and y_pred.shape[1] == 2, "y_pred must be (N,2)"
	return float(np.mean(y_pred[:, 1] - y_pred[:, 0]))


def compute_coverage_width_criterion(
	y_pred: np.ndarray,
	y_obs: np.ndarray,
	alpha: float = 0.5,
	lam: float = 10.0,
	eta: float = 50.0,
	include_endpoints: bool = True,
) -> float:
	"""CWC: Coverage-Width-based Criterion.

	CWC = MPIW * (1 + lam * exp(-eta * (PICP - (1 - alpha))))^gamma,
	where gamma = 1 if PICP < (1 - alpha) else 0.

	Requires scalar observations y_obs for coverage.
	"""
	picp = compute_prediction_interval_coverage_probability(y_pred, y_obs, include_endpoints=include_endpoints)
	mpiw = compute_mean_prediction_interval_width(y_pred)
	gamma = 1.0 if picp < (1.0 - alpha) else 0.0
	return float(mpiw * ((1.0 + lam * np.exp(-eta * (picp - (1.0 - alpha)))) ** gamma))


def compute_overlap_hits(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	tolerance: float = 0.0,
	tolerance_mode: str = 'relative_to_true_width',
	include_endpoints: bool = False,
) -> np.ndarray:
	"""Return per-sample overlap indicators (1 if overlap, else 0).

	If tolerance > 0, expand the true interval by the specified mode before checking overlap.
	"""
	assert y_true.shape == y_pred.shape and y_true.shape[1] == 2, "Intervals must be (N,2)"
	N = y_true.shape[0]
	hits = np.zeros(N, dtype=float)
	for i in range(N):
		amin, amax = float(y_true[i, 0]), float(y_true[i, 1])
		pmin, pmax = float(y_pred[i, 0]), float(y_pred[i, 1])
		if tolerance > 0.0:
			if tolerance_mode == 'relative_to_true_width':
				w = max(0.0, amax - amin)
				rmin, rmax = amin - tolerance*w, amax + tolerance*w
			elif tolerance_mode == 'scale_endpoints':
				rmin, rmax = amin*(1.0 - tolerance), amax*(1.0 + tolerance)
			else:
				raise ValueError("tolerance_mode must be 'relative_to_true_width' or 'scale_endpoints'")
		else:
			rmin, rmax = amin, amax
		inter = min(pmax, rmax) - max(pmin, rmin)
		hits[i] = 1.0 if (inter >= 0.0 if include_endpoints else inter > 0.0) else 0.0
	return hits


def _per_sample_width_error_normalized(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	width_normalization: str = 'dataset_mean',
) -> np.ndarray:
	"""Per-sample normalized width error consistent with compute_interval_quality_score.

	- dataset_mean: |pred_w - true_w| / mean(true_w)
	- by_true_width: |pred_w - true_w| / true_w (per-sample)
	"""
	assert y_true.shape == y_pred.shape and y_true.shape[1] == 2, "Intervals must be (N,2)"
	true_w = y_true[:, 1] - y_true[:, 0]
	pred_w = y_pred[:, 1] - y_pred[:, 0]
	width_err = np.abs(pred_w - true_w)
	if width_normalization == 'dataset_mean':
		denom = float(np.mean(np.maximum(true_w, 1e-12)))
		return width_err / max(denom, 1e-12)
	elif width_normalization == 'by_true_width':
		return width_err / np.maximum(true_w, 1e-12)
	else:
		raise ValueError("width_normalization must be 'dataset_mean' or 'by_true_width'")


def compute_balanced_lambda(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	width_normalization: str = 'dataset_mean',
	tolerance: float = 0.0,
	tolerance_mode: str = 'relative_to_true_width',
	use_median: bool = False,
) -> float:
	"""Data-driven lambda to balance average contributions of (1-LOA) and MAE_width_norm.

	Returns
	-------
	float
		lambda_bal = E[1-LOA] / E[MAE_width_norm]; with medians if use_median=True.
	"""
	hits = compute_overlap_hits(y_true, y_pred, tolerance=tolerance, tolerance_mode=tolerance_mode, include_endpoints=False)
	miss = 1.0 - hits
	width_err_norm = _per_sample_width_error_normalized(y_true, y_pred, width_normalization=width_normalization)
	if use_median:
		num = float(np.median(miss))
		den = float(np.median(width_err_norm))
	else:
		num = float(np.mean(miss))
		den = float(np.mean(width_err_norm))
	return float(num / max(den, 1e-12))


def compute_asymmetric_miss_score(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	alpha: float = 1.0,
	return_per_sample: bool = False,
) -> float | np.ndarray:
	"""Asymmetric Miss Score (AMS).

	AMS = (alpha * UnderError + OverError) / TrueWidth,
	where UnderError = TrueWidth - IntersectionLength,
	      OverError  = PredWidth  - IntersectionLength.

	- IntersectionLength is max(0, min(a_max,p_max) - max(a_min,p_min)).
	- TrueWidth = a_max - a_min (clipped at 1e-12 for stability).
	- PredWidth = p_max - p_min after ordering endpoints if needed.
	- Non-negative and scale-invariant; alpha > 1 penalizes misses harder than waste.

	Parameters
	----------
	y_true : np.ndarray of shape (N,2)
	y_pred : np.ndarray of shape (N,2)
	alpha : float
		Asymmetry weight for under-prediction vs over-prediction.
	return_per_sample : bool
		If True, return per-sample AMS array; otherwise return the mean.
	"""
	assert y_true.shape == y_pred.shape and y_true.shape[1] == 2, "Intervals must be (N,2)"
	assert alpha >= 0.0, "alpha must be non-negative"
	# Ensure ordering of endpoints
	Tmin = np.minimum(y_true[:, 0], y_true[:, 1])
	Tmax = np.maximum(y_true[:, 0], y_true[:, 1])
	Pmin = np.minimum(y_pred[:, 0], y_pred[:, 1])
	Pmax = np.maximum(y_pred[:, 0], y_pred[:, 1])
	true_w = np.maximum(Tmax - Tmin, 1e-12)
	pred_w = np.maximum(Pmax - Pmin, 0.0)
	inter = np.maximum(0.0, np.minimum(Tmax, Pmax) - np.maximum(Tmin, Pmin))
	under_err = np.maximum(0.0, true_w - inter)
	over_err = np.maximum(0.0, pred_w - inter)
	ams = (alpha * under_err + over_err) / true_w
	return ams if return_per_sample else float(np.mean(ams))


# ------------------------------
# CLI visualizations for paper figures (toy examples)
# ------------------------------

def _case1_interval_contrast(output_dir: Path):
	true_intervals = np.array([[100, 200], [100, 200]])
	pred_low_mae_no_overlap = np.array([[201, 220], [80, 95]])
	pred_high_mae_overlap = np.array([[190, 310], [90, 120]])

	mae_no = mean_absolute_error(true_intervals.ravel(), pred_low_mae_no_overlap.ravel())
	mae_ov = mean_absolute_error(true_intervals.ravel(), pred_high_mae_overlap.ravel())
	ov_no = lenient_overlap_accuracy(true_intervals, pred_low_mae_no_overlap)
	ov_ov = lenient_overlap_accuracy(true_intervals, pred_high_mae_overlap)
	print(f"Case1: Low-MAE, No-Overlap -> MAE={mae_no:.2f}, Overlap={ov_no:.2f}")
	print(f"Case1: Higher-MAE, Overlap -> MAE={mae_ov:.2f}, OverlapS={ov_ov:.2f}")

	fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
	for i, (ti, pi) in enumerate(zip(true_intervals, pred_low_mae_no_overlap)):
		ax[0].plot(ti, [i, i], 'k-', lw=6, label='true' if i==0 else None)
		ax[0].plot(pi, [i+0.1, i+0.1], 'r-', lw=4, label='pred' if i==0 else None)
	ax[0].set_title('Low MAE, No Overlap')
	ax[0].legend()
	for i, (ti, pi) in enumerate(zip(true_intervals, pred_high_mae_overlap)):
		ax[1].plot(ti, [i, i], 'k-', lw=6, label='true' if i==0 else None)
		ax[1].plot(pi, [i+0.1, i+0.1], 'g-', lw=4, label='pred' if i==0 else None)
	ax[1].set_title('Higher MAE, Overlap')
	plt.tight_layout()
	fig_path = output_dir / 'case1_intervals.png'
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _case2_scatter_mae_overlap(output_dir: Path, seed: int = 42):
	rng = np.random.default_rng(seed)
	n = 200
	low = rng.uniform(0, 1000, size=n)
	high = low + rng.uniform(50, 300, size=n)
	y_true = np.vstack([low, high]).T
	pred_low = low + rng.normal(0, 120, size=n)
	pred_high = high + rng.normal(0, 120, size=n)
	y_pred = np.vstack([pred_low, pred_high]).T
	swap = y_pred[:, 1] < y_pred[:, 0]
	y_pred[swap, :] = y_pred[swap, ::-1]
	mae = np.mean(np.abs(y_true - y_pred), axis=1)
	overlap = np.array([
		1.0 if max(0, min(y_true[i,1], y_pred[i,1]) - max(y_true[i,0], y_pred[i,0])) > 0 else 0.0
		for i in range(n)
	])
	fig, ax = plt.subplots(figsize=(5,4))
	ax.scatter(mae, overlap, alpha=0.5)
	ax.set_xlabel('Endpoint MAE')
	ax.set_ylabel('Overlap (0/1)')
	ax.set_title('MAE vs Overlap')
	ax.set_yticks([0,1])
	ax.grid(True, ls=':')
	fig_path = output_dir / 'case2_scatter_mae_overlap.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _case3_coverage_tolerance(output_dir: Path):
	true = np.array([[100, 200], [400, 600], [50, 70], [1000, 1200]])
	pred = np.array([[190, 310], [350, 580], [60, 80], [800, 900]])
	def relaxed_overlap(y_true, y_pred, tol):
		matches = []
		for i in range(len(y_true)):
			a_min, a_max = y_true[i]
			p_min, p_max = y_pred[i]
			rmin, rmax = a_min*(1-tol), a_max*(1+tol)
			inter = max(0, min(p_max, rmax) - max(p_min, rmin))
			matches.append(inter > 0)
		return np.mean(matches)
	T = np.linspace(0.0, 0.3, 7)
	cover = [relaxed_overlap(true, pred, t) for t in T]
	fig, ax = plt.subplots(figsize=(5,3))
	ax.plot(T, cover, marker='o')
	ax.set_xlabel('Tolerance')
	ax.set_ylabel('Coverage')
	ax.set_title('Coverage vs Tolerance')
	ax.grid(True, ls=':')
	fig_path = output_dir / 'case3_coverage_tolerance.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _case4_strict_vs_mape(output_dir: Path):
	# Overconfident underprediction
	y_true_a = np.array([10.0, 1000.0])
	y_pred_a = np.array([1.0, 900.0])
	se_a = strict_error(y_true_a, y_pred_a)
	mape_a = mean_absolute_percentage_error(y_true_a, y_pred_a)
	# Similar scale
	y_true_b = np.array([1000.0, 2000.0, 3000.0])
	y_pred_b = np.array([900.0, 2200.0, 2700.0])
	se_b = strict_error(y_true_b, y_pred_b)
	mape_b = mean_absolute_percentage_error(y_true_b, y_pred_b)
	print(f"Case4: A strict={se_a:.3f} vs MAPE={mape_a:.3f}; B strict={se_b:.3f} vs MAPE={mape_b:.3f}")
	fig, ax = plt.subplots(figsize=(5,3))
	ax.bar(['A_strict','A_MAPE','B_strict','B_MAPE'], [se_a, mape_a, se_b, mape_b], color=['C0','C1','C0','C1'])
	ax.set_title('Strict vs MAPE Sensitivity')
	fig_path = output_dir / 'case4_strict_vs_mape.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _case5_auroc_overlap_vs_mae(output_dir: Path, seed: int = 42):
	rng = np.random.default_rng(seed)
	n = 300
	low = rng.uniform(0, 1000, size=n)
	high = low + rng.uniform(50, 300, size=n)
	y_true = np.vstack([low, high]).T
	pred_low = low + rng.normal(0, 120, size=n)
	pred_high = high + rng.normal(0, 120, size=n)
	y_pred = np.vstack([pred_low, pred_high]).T
	swap = y_pred[:, 1] < y_pred[:, 0]
	y_pred[swap, :] = y_pred[swap, ::-1]
	mae = np.mean(np.abs(y_true - y_pred), axis=1)
	overlap = np.array([
		1.0 if max(0, min(y_true[i,1], y_pred[i,1]) - max(y_true[i,0], y_pred[i,0])) > 0 else 0.0
		for i in range(n)
	])
	labels = (overlap > 0).astype(int)
	x_overlap = overlap
	x_mae_inv = 1.0 / (1.0 + mae)
	auroc_overlap = roc_auc_score(labels, x_overlap)
	aupr_overlap = average_precision_score(labels, x_overlap)
	auroc_maeinv = roc_auc_score(labels, x_mae_inv)
	aupr_maeinv = average_precision_score(labels, x_mae_inv)
	print(f"Case5 AUROC overlap={auroc_overlap:.3f} vs MAE-inv={auroc_maeinv:.3f}")
	print(f"Case5 AUPR  overlap={aupr_overlap:.3f} vs MAE-inv={aupr_maeinv:.3f}")
	fig, ax = plt.subplots(figsize=(5,3))
	ax.bar(['AUROC_overlap','AUROC_MAEinv','AUPR_overlap','AUPR_MAEinv'], [auroc_overlap, auroc_maeinv, aupr_overlap, aupr_maeinv])
	ax.set_ylim(0,1)
	fig_path = output_dir / 'case5_auroc_overlap_vs_mae.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


# ------------------------------
# Extended visualizations (case6–case10)
# ------------------------------

def _synth_featured_data(n: int = 2000, seed: int = 123):
	"""Synthesize intervals and features (concurrency, metadata_ratio) with overlap structure."""
	rng = np.random.default_rng(seed)
	concurrency = rng.uniform(0.0, 1.0, size=n)
	metadata_ratio = rng.beta(2.0, 5.0, size=n)  # skewed toward low
	X = np.column_stack([concurrency, metadata_ratio])
	# True interval centers/widths depend on features
	true_center = 500 + 400*concurrency - 300*metadata_ratio
	true_width = 80 + 200*metadata_ratio + 50*rng.standard_normal(size=n)
	true_width = np.clip(true_width, 40, 400)
	y_true = np.column_stack([true_center - true_width/2, true_center + true_width/2])
	# Predicted interval with systematic bias and noise
	pred_center = true_center + 60*(metadata_ratio - 0.3) + 50*rng.standard_normal(size=n)
	pred_width = true_width*(0.9 + 0.4*rng.random(size=n))
	y_pred = np.column_stack([pred_center - pred_width/2, pred_center + pred_width/2])
	# Ensure proper ordering
	swap = y_pred[:, 1] < y_pred[:, 0]
	y_pred[swap, :] = y_pred[swap, ::-1]
	return X, y_true, y_pred


def _interval_iou(a: np.ndarray, b: np.ndarray) -> float:
	"""IoU for 1D intervals a=[amin,amax], b=[bmin,bmax]."""
	inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
	union = max(a[1], b[1]) - min(a[0], b[0])
	return 0.0 if union <= 0 else inter/union


def _case6_coverage_heatmap(output_dir: Path, bins: int = 30, seed: int = 123):
	X, y_true, y_pred = _synth_featured_data(seed=seed)
	concurrency = X[:,0]
	metadata_ratio = X[:,1]
	# Compute overlap label
	overlap = np.array([
		1.0 if max(0.0, min(y_true[i,1], y_pred[i,1]) - max(y_true[i,0], y_pred[i,0])) > 0 else 0.0
		for i in range(len(X))
	])
	# Bin heatmap
	xbins = np.linspace(0, 1, bins+1)
	ybins = np.linspace(0, 1, bins+1)
	grid = np.zeros((bins, bins))
	counts = np.zeros((bins, bins))
	for xi, yi, ov in zip(concurrency, metadata_ratio, overlap):
		xi_bin = min(bins-1, np.searchsorted(xbins, xi, side='right')-1)
		yi_bin = min(bins-1, np.searchsorted(ybins, yi, side='right')-1)
		grid[yi_bin, xi_bin] += ov
		counts[yi_bin, xi_bin] += 1
	with np.errstate(invalid='ignore'):
		cov = np.divide(grid, counts, out=np.zeros_like(grid), where=counts>0)
	fig, ax = plt.subplots(figsize=(5,4))
	im = ax.imshow(cov, origin='lower', extent=[0,1,0,1], aspect='auto', vmin=0, vmax=1, cmap='viridis')
	plt.colorbar(im, ax=ax, label='Overlap rate')
	ax.set_xlabel('concurrency')
	ax.set_ylabel('metadata_ratio')
	ax.set_title('Coverage heatmap (overlap rate)')
	fig_path = output_dir / 'case6_coverage_heatmap.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _case7_iou_vs_mae_density(output_dir: Path, seed: int = 123):
	X, y_true, y_pred = _synth_featured_data(seed=seed)
	iou = np.array([_interval_iou(y_true[i], y_pred[i]) for i in range(len(y_true))])
	mae = np.mean(np.abs(y_true - y_pred), axis=1)
	fig, ax = plt.subplots(figsize=(5,4))
	hb = ax.hexbin(mae, iou, gridsize=40, cmap='magma', extent=[0, np.percentile(mae,95), 0,1])
	plt.colorbar(hb, ax=ax, label='count')
	ax.set_xlabel('Endpoint MAE')
	ax.set_ylabel('IoU (interval)')
	ax.set_title('IoU vs MAE (density)')
	fig_path = output_dir / 'case7_iou_vs_mae_density.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 300, alpha: float = 0.05, seed: int = 0):
	rng = np.random.default_rng(seed)
	boot = []
	n = len(vals)
	for _ in range(n_boot):
		idx = rng.integers(0, n, size=n)
		boot.append(np.mean(vals[idx]))
	lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
	return lo, hi


def _case8_calibration_tolerance_ci(output_dir: Path, seed: int = 123):
	_, y_true, y_pred = _synth_featured_data(seed=seed)
	def relaxed_match(tol):
		arr = []
		for i in range(len(y_true)):
			a_min, a_max = y_true[i]
			p_min, p_max = y_pred[i]
			rmin, rmax = a_min*(1-tol), a_max*(1+tol)
			inter = max(0, min(p_max, rmax) - max(p_min, rmin))
			arr.append(inter > 0)
		return np.array(arr, dtype=float)
	T = np.linspace(0.0, 0.3, 7)
	cover = []
	lo_ci = []
	hi_ci = []
	for t in T:
		m = relaxed_match(t)
		cover.append(np.mean(m))
		lo, hi = _bootstrap_ci(m)
		lo_ci.append(lo)
		hi_ci.append(hi)
	fig, ax = plt.subplots(figsize=(5,3))
	ax.plot(T, cover, marker='o')
	ax.fill_between(T, lo_ci, hi_ci, alpha=0.2, color='C0')
	ax.set_xlabel('Tolerance')
	ax.set_ylabel('Coverage')
	ax.set_title('Coverage vs tolerance with 95% CI')
	ax.grid(True, ls=':')
	fig_path = output_dir / 'case8_calibration_tolerance_ci.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def _vif_table(X: np.ndarray, feature_names: List[str]) -> List[tuple]:
	"""Compute VIF using linear regression R^2 per feature."""
	rows = []
	for i, name in enumerate(feature_names):
		X_i = X[:, i]
		X_others = np.delete(X, i, axis=1)
		if X_others.shape[1] == 0:
			vif = 1.0
		else:
			lr = LinearRegression().fit(X_others, X_i)
			r2 = lr.score(X_others, X_i)
			vif = np.inf if r2 >= 0.9999 else 1.0/(1.0 - r2)
		rows.append((name, vif))
	return rows


def _case9_logistic_pd_vif(output_dir: Path, seed: int = 123):
	X, y_true, y_pred = _synth_featured_data(seed=seed)
	y = np.array([
		1 if max(0.0, min(y_true[i,1], y_pred[i,1]) - max(y_true[i,0], y_pred[i,0])) > 0 else 0
		for i in range(len(X))
	])
	clf = LogisticRegression(max_iter=200).fit(X, y)
	# 1D PD for each feature
	fnames = ['concurrency','metadata_ratio']
	fig, ax = plt.subplots(1,2, figsize=(9,3))
	for j in range(2):
		grid = np.linspace(0,1,50)
		X_ref = np.tile(np.median(X, axis=0), (50,1))
		X_ref[:, j] = grid
		proba = clf.predict_proba(X_ref)[:,1]
		ax[j].plot(grid, proba)
		ax[j].set_xlabel(fnames[j])
		ax[j].set_ylabel('P(overlap)')
		ax[j].set_title(f'PD: {fnames[j]}')
	fig_path = output_dir / 'case9_logistic_pd.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")
	# 2D contour of predicted prob
	gx = gy = np.linspace(0,1,60)
	GX, GY = np.meshgrid(gx, gy)
	grid2 = np.column_stack([GX.ravel(), GY.ravel()])
	P = clf.predict_proba(grid2)[:,1].reshape(GX.shape)
	fig, ax = plt.subplots(figsize=(5,4))
	cs = ax.contourf(GX, GY, P, levels=20, cmap='viridis')
	plt.colorbar(cs, ax=ax, label='P(overlap)')
	ax.set_xlabel('concurrency')
	ax.set_ylabel('metadata_ratio')
	ax.set_title('Logistic P(overlap) contour')
	fig_path = output_dir / 'case9_logistic_contour.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")
	# VIF table (save as txt)
	vifs = _vif_table(X, fnames)
	vif_path = output_dir / 'case9_vif.txt'
	with open(vif_path, 'w') as f:
		for name, vif in vifs:
			f.write(f"{name}: VIF={vif:.2f}\n")
	print(f"Saved: {vif_path}")


def _case10_reliability_nominal_empirical(output_dir: Path, seed: int = 123):
	_, y_true, y_pred = _synth_featured_data(seed=seed)
	# Scale predicted widths to emulate different nominal coverages
	nominals = [0.5, 0.8, 0.9]
	centers = (y_pred[:,0] + y_pred[:,1]) / 2
	base_half = (y_pred[:,1] - y_pred[:,0]) / 2
	empirical = []
	for nom in nominals:
		scale = nom / 0.5  # assume current approx 50% coverage baseline
		low = centers - base_half*scale
		high = centers + base_half*scale
		m = [max(0, min(high[i], y_true[i,1]) - max(low[i], y_true[i,0])) > 0 for i in range(len(y_true))]
		empirical.append(np.mean(m))
	fig, ax = plt.subplots(figsize=(4,4))
	ax.plot(nominals, empirical, marker='o')
	ax.plot([0,1],[0,1],'k--', alpha=0.5)
	ax.set_xlim(0.45, 0.95)
	ax.set_ylim(0,1)
	ax.set_xlabel('Nominal coverage')
	ax.set_ylabel('Empirical coverage')
	ax.set_title('Reliability diagram (intervals)')
	fig_path = output_dir / 'case10_reliability.png'
	plt.tight_layout()
	plt.savefig(fig_path, dpi=200)
	plt.close(fig)
	print(f"Saved: {fig_path}")


def main():
	parser = argparse.ArgumentParser(description='Metrics visualization CLI (toy examples)')
	parser.add_argument('--case', choices=['case1','case2','case3','case4','case5','case6','case7','case8','case9','case10','all'], default='all')
	parser.add_argument('--output-dir', type=Path, default=Path.cwd())
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	args.output_dir.mkdir(parents=True, exist_ok=True)

	if args.case in ('case1','all'):
		_case1_interval_contrast(args.output_dir)
	if args.case in ('case2','all'):
		_case2_scatter_mae_overlap(args.output_dir, seed=args.seed)
	if args.case in ('case3','all'):
		_case3_coverage_tolerance(args.output_dir)
	if args.case in ('case4','all'):
		_case4_strict_vs_mape(args.output_dir)
	if args.case in ('case5','all'):
		_case5_auroc_overlap_vs_mae(args.output_dir, seed=args.seed)
	if args.case in ('case6','all'):
		_case6_coverage_heatmap(args.output_dir)
	if args.case in ('case7','all'):
		_case7_iou_vs_mae_density(args.output_dir)
	if args.case in ('case8','all'):
		_case8_calibration_tolerance_ci(args.output_dir)
	if args.case in ('case9','all'):
		_case9_logistic_pd_vif(args.output_dir)
	if args.case in ('case10','all'):
		_case10_reliability_nominal_empirical(args.output_dir)


if __name__ == '__main__':
	main()