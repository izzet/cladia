import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from dfdiagnoser_ml.metrics import (
    lenient_overlap_accuracy,
    strict_error,
    lenient_overlap_accuracy_tolerant,
    compute_interval_quality_score,
    compute_winkler_score,
    compute_mean_in_interval_rate,
    compute_prediction_interval_coverage_probability,
    compute_asymmetric_miss_score,
)


def test_lenient_overlap_detects_coverage_even_with_large_endpoint_error():
    # Two intervals with clear overlap but endpoints are far from true endpoints
    # True intervals
    y_true = np.array([
        [100.0, 200.0],
        [100.0, 200.0],
    ])
    # Predicted intervals: both overlap but endpoints have large absolute errors
    y_pred = np.array([
        [190.0, 310.0],   # overlaps with [100,200] on [190,200]
        [90.0, 120.0],    # overlaps with [100,200] on [100,120]
    ])

    # Endpoint MAE can be large
    mae_min = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_max = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    overlap = lenient_overlap_accuracy(y_true, y_pred)

    assert overlap == 1.0, "All intervals overlap; coverage should be 1.0"
    assert mae_min > 0 or mae_max > 0, "Endpoint errors exist despite full overlap"


def test_lenient_overlap_zero_despite_small_mae():
    # Intervals are close but do not overlap
    y_true = np.array([[100.0, 101.0]])
    y_pred = np.array([[101.1, 102.0]])   # Small endpoint error but no overlap

    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    overlap = lenient_overlap_accuracy(y_true, y_pred)

    assert mae < 2.0, "Endpoints are numerically close"
    assert overlap == 0.0, "No overlap should yield 0.0 coverage"


def test_strict_error_penalizes_overconfident_underprediction_more_than_mape():
    # strict_error normalizes by |y_pred|, so overconfident small predictions are heavily penalized
    y_true = np.array([10.0, 1000.0])
    y_pred = np.array([1.0, 900.0])  # first case: severe underprediction; second: mild underprediction

    se = strict_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # In this mixed case, strict_error should exceed MAPE due to the first term's |y_pred|-denominator
    assert se > mape, "Strict error should penalize overconfident small predictions more than MAPE"


def test_strict_error_and_mape_behave_similarly_when_pred_magnitudes_are_large():
    # When predictions are of similar magnitude to truths, strict and MAPE become closer
    y_true = np.array([1000.0, 2000.0, 3000.0])
    y_pred = np.array([900.0, 2200.0, 2700.0])

    se = strict_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Not necessarily equal, but should be in the same ballpark
    assert abs(se - mape) < 0.2, "With large predictions, strict and MAPE should be closer in magnitude"



def test_iqs_and_winkler_scenarios_A_to_D_with_corrected_values():
    # Common true interval
    y_true = np.array([[9.8, 11.2]])
    true_width = 1.4
    y_point = np.array([10.5])

    # Scenario A: Good fit [9.7, 11.3]
    pred_A = np.array([[9.7, 11.3]])
    winkler_A = compute_winkler_score(pred_A, y_point, alpha=0.5, include_endpoints=True)
    iqs_A_strict = compute_interval_quality_score(y_true, pred_A, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    assert abs(winkler_A - 1.6) < 1e-8
    assert abs(iqs_A_strict - (0.2/true_width)) < 1e-8  # 0.142857...

    # Scenario B: Shifted [11.3, 12.7]
    pred_B = np.array([[11.3, 12.7]])
    winkler_B = compute_winkler_score(pred_B, y_point, alpha=0.5, include_endpoints=True)
    iqs_B_strict = compute_interval_quality_score(y_true, pred_B, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    # Tolerant IQS with small tolerance relative to true width should count as overlap and width error is zero
    iqs_B_tol = compute_interval_quality_score(y_true, pred_B, lambda_weight=1.0, tolerance=0.1, width_normalization='dataset_mean', tolerance_mode='relative_to_true_width')
    assert abs(winkler_B - 4.6) < 1e-8
    assert abs(iqs_B_strict - 1.0) < 1e-8
    assert abs(iqs_B_tol - 0.0) < 1e-8

    # Scenario C: Too wide [8.0, 13.0]
    pred_C = np.array([[8.0, 13.0]])
    winkler_C = compute_winkler_score(pred_C, y_point, alpha=0.5, include_endpoints=True)
    iqs_C = compute_interval_quality_score(y_true, pred_C, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    assert abs(winkler_C - 5.0) < 1e-8
    assert abs(iqs_C - ((5.0 - true_width)/true_width)) < 1e-8  # 2.571428...

    # Scenario D: Too narrow [10.0, 10.5]
    pred_D = np.array([[10.0, 10.5]])
    winkler_D = compute_winkler_score(pred_D, y_point, alpha=0.5, include_endpoints=True)
    iqs_D = compute_interval_quality_score(y_true, pred_D, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    assert abs(winkler_D - 0.5) < 1e-8
    assert abs(iqs_D - ((true_width - 0.5)/true_width)) < 1e-8  # 0.642857...


def test_additional_scenarios_iqs_superior_to_winkler():
    # Scenario 1: Correctly Uncertain
    y_true1 = np.array([[10.0, 30.0]])
    pred1 = np.array([[11.0, 29.0]])
    y_point1 = np.array([20.0])
    winkler1 = compute_winkler_score(pred1, y_point1, alpha=0.5, include_endpoints=True)
    iqs1 = compute_interval_quality_score(y_true1, pred1, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    assert abs(winkler1 - 18.0) < 1e-8
    assert abs(iqs1 - 0.1) < 1e-8

    # Scenario 2: Skewed Distribution (mean outside IQR)
    y_true2 = np.array([[8.0, 12.0]])
    pred2 = np.array([[7.5, 11.5]])
    y_point2 = np.array([12.5])
    winkler2 = compute_winkler_score(pred2, y_point2, alpha=0.5, include_endpoints=True)
    iqs2 = compute_interval_quality_score(y_true2, pred2, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    assert abs(winkler2 - 8.0) < 1e-8
    assert abs(iqs2 - 0.0) < 1e-8

    # Scenario 3: Over-Narrow & Centered
    y_true3 = np.array([[9.8, 11.2]])
    pred3 = np.array([[10.4, 10.6]])
    y_point3 = np.array([10.5])
    winkler3 = compute_winkler_score(pred3, y_point3, alpha=0.5, include_endpoints=True)
    iqs3 = compute_interval_quality_score(y_true3, pred3, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    assert abs(winkler3 - 0.2) < 1e-8
    assert abs(iqs3 - (abs(0.2 - 1.4)/1.4)) < 1e-8  # 0.857142...

    # Scenario 4: Touching intervals (strict miss, practical hit with small tolerance)
    y_true4 = np.array([[9.8, 11.2]])
    pred4 = np.array([[11.2, 12.6]])
    y_point4 = np.array([10.5])
    winkler4 = compute_winkler_score(pred4, y_point4, alpha=0.5, include_endpoints=True)
    iqs4_strict = compute_interval_quality_score(y_true4, pred4, lambda_weight=1.0, tolerance=0.0, width_normalization='dataset_mean')
    iqs4_tol = compute_interval_quality_score(y_true4, pred4, lambda_weight=1.0, tolerance=0.05, width_normalization='dataset_mean', tolerance_mode='relative_to_true_width')
    assert abs(winkler4 - (1.4 + 2.8)) < 1e-8  # 4.2
    assert abs(iqs4_strict - 1.0) < 1e-8
    assert abs(iqs4_tol - 0.0) < 1e-8


def test_mir_and_picp_behave_as_expected_in_scenarios():
    # Correctly Uncertain: design 10 obs with 5 inside [10,30]; pred [11,29] should also cover 5
    y_obs1 = np.array([5, 9, 11, 15, 20, 25, 29, 31, 35, 40])
    pred1 = np.array([[11.0, 29.0]])
    picp1 = compute_prediction_interval_coverage_probability(pred1, y_obs1, include_endpoints=True)
    assert abs(picp1 - 0.5) < 1e-8

    # Skewed Distribution: 5 inside [7.5,11.5]
    y_obs2 = np.array([6.0, 7.0, 8.5, 9.0, 10.0, 11.0, 11.5, 12.5, 14.0, 16.0])
    pred2 = np.array([[7.5, 11.5]])
    picp2 = compute_prediction_interval_coverage_probability(pred2, y_obs2, include_endpoints=True)
    assert abs(picp2 - 0.5) < 1e-8

    # Over-Narrow & Centered: only 2 inside [10.4,10.6]; MIR should be 100% because mean 10.5 is inside
    y_obs3 = np.array([9.0, 9.9, 10.2, 10.45, 10.55, 10.8, 11.0, 11.3, 12.0, 13.0])
    pred3 = np.array([[10.4, 10.6]])
    picp3 = compute_prediction_interval_coverage_probability(pred3, y_obs3, include_endpoints=True)
    y_point3 = np.array([10.5])
    mir3 = compute_mean_in_interval_rate(pred3, y_point3, include_endpoints=True)
    assert abs(picp3 - 0.2) < 1e-8
    assert abs(mir3 - 1.0) < 1e-8

    # Touching intervals: design 10 obs with exactly 2 inside [11.2,12.6]
    y_obs4 = np.array([9.0, 9.5, 9.8, 10.0, 10.3, 10.7, 11.2, 12.4, 12.7, 13.5])
    pred4 = np.array([[11.2, 12.6]])
    picp4 = compute_prediction_interval_coverage_probability(pred4, y_obs4, include_endpoints=True)
    assert abs(picp4 - 0.2) < 1e-8


def test_balanced_lambda_estimation_and_ranking_stability():
    # Construct a small synthetic set of intervals with varying behaviors
    y_true = np.array([
        [100.0, 200.0],  # typical
        [50.0, 70.0],    # narrow
        [400.0, 600.0],  # wide
        [1000.0, 1200.0] # wide
    ])
    # Predictions: one good overlap, one over-narrow, one shifted, one too-wide
    y_pred = np.array([
        [110.0, 190.0],   # overlaps, slight width error
        [58.0, 60.0],     # over-narrow inside
        [1210.0, 1310.0], # shifted away from [1000,1200]
        [300.0, 900.0],   # too-wide covering
    ])

    # Compute balanced lambda using means and medians
    from dfdiagnoser_ml.metrics import compute_balanced_lambda, compute_interval_quality_score
    lam_mean = compute_balanced_lambda(y_true, y_pred, width_normalization='dataset_mean', tolerance=0.0, use_median=False)
    lam_median = compute_balanced_lambda(y_true, y_pred, width_normalization='dataset_mean', tolerance=0.0, use_median=True)

    assert lam_mean > 0
    assert lam_median > 0

    # Ranking stability across a small lambda grid
    lambdas = [0.5, 1.0, 2.0]
    scores = []
    for lam in lambdas:
        scores.append(compute_interval_quality_score(y_true, y_pred, lambda_weight=lam, tolerance=0.0, width_normalization='dataset_mean'))

    # As lambda increases, the score should monotonically increase or stay same because width penalty grows (LOA fixed here)
    assert scores[0] <= scores[1] <= scores[2]


def test_ams_basic_cases_and_properties():
    # Perfect match => AMS = 0
    y_true = np.array([[10.0, 20.0]])
    y_pred = np.array([[10.0, 20.0]])
    ams = compute_asymmetric_miss_score(y_true, y_pred, alpha=2.0)
    assert abs(ams - 0.0) < 1e-12

    # Predicted subset inside true (under-coverage=0, over=0) => AMS=0
    y_pred2 = np.array([[12.0, 18.0]])
    ams2 = compute_asymmetric_miss_score(y_true, y_pred2, alpha=2.0)
    # Intersection=6, true_w=10, pred_w=6; under=true_w-inter=4, over=pred-inter=0 => AMS=(2*4+0)/10=0.8
    # NOTE: Under this definition, a subset that leaves part of true uncovered is penalized; this is intended.
    assert abs(ams2 - 0.8) < 1e-12

    # Predicted superset (covers true but wider): under=0, over>0
    y_pred3 = np.array([[8.0, 22.0]])
    ams3 = compute_asymmetric_miss_score(y_true, y_pred3, alpha=2.0)
    # inter=10, pred_w=14 => over=4; AMS=(0+4)/10=0.4
    assert abs(ams3 - 0.4) < 1e-12

    # No overlap
    y_pred4 = np.array([[30.0, 40.0]])
    ams4 = compute_asymmetric_miss_score(y_true, y_pred4, alpha=2.0)
    # inter=0, under=10, over=10 => AMS=(2*10 + 10)/10 = 3.0
    assert abs(ams4 - 3.0) < 1e-12

    # Touching (zero-length intersection) should count as no overlap in AMS
    y_pred5 = np.array([[20.0, 25.0]])
    ams5 = compute_asymmetric_miss_score(y_true, y_pred5, alpha=1.0)
    # inter=0, under=10, over=5 => AMS=(1*10+5)/10=1.5
    assert abs(ams5 - 1.5) < 1e-12

    # Monotonicity in alpha: increasing alpha increases AMS when under_err>0
    y_pred6 = np.array([[12.0, 18.0]])
    ams_a1 = compute_asymmetric_miss_score(y_true, y_pred6, alpha=1.0)
    ams_a2 = compute_asymmetric_miss_score(y_true, y_pred6, alpha=2.0)
    assert ams_a2 > ams_a1

    # Scale invariance: scaling intervals by c leaves AMS unchanged
    c = 5.0
    y_true_scaled = y_true * c
    y_pred_scaled = y_pred3 * c
    ams_scaled = compute_asymmetric_miss_score(y_true_scaled, y_pred_scaled, alpha=2.0)
    assert abs(ams_scaled - ams3) < 1e-12
