import numpy as np
import pandas as pd

from dfdiagnoser_ml.common import add_compute_time_frac_epoch_quantiles


def make_df(mu_x, sd_x, mu_y, sd_y):
    return pd.DataFrame({
        "compute_time_q25_q75_mean": [mu_x],
        "compute_time_q25_q75_std": [sd_x],
        "epoch_time_q25_q75_mean": [mu_y],
        "epoch_time_q25_q75_std": [sd_y],
    })


def test_delta_method_monotonicity():
    # If compute increases relative to epoch (same sd), ratio quantiles should increase
    df1 = make_df(mu_x=5.0, sd_x=0.5, mu_y=10.0, sd_y=0.5)
    df2 = make_df(mu_x=6.0, sd_x=0.5, mu_y=10.0, sd_y=0.5)
    r1 = add_compute_time_frac_epoch_quantiles(df1, method="delta")
    r2 = add_compute_time_frac_epoch_quantiles(df2, method="delta")
    assert r2.loc[0, "compute_time_frac_epoch_q25"] > r1.loc[0, "compute_time_frac_epoch_q25"]
    assert r2.loc[0, "compute_time_frac_epoch_q75"] > r1.loc[0, "compute_time_frac_epoch_q75"]


def test_delta_method_bounds():
    # With small sd and mu_x < mu_y, ratio q75 should be < 1 and q25 > 0
    df = make_df(mu_x=2.0, sd_x=0.1, mu_y=8.0, sd_y=0.2)
    r = add_compute_time_frac_epoch_quantiles(df, method="delta")
    q25 = float(r.loc[0, "compute_time_frac_epoch_q25"])
    q75 = float(r.loc[0, "compute_time_frac_epoch_q75"])
    assert 0.0 <= q25 < 1.0
    assert 0.0 < q75 < 1.0
    assert q75 > q25


def test_mc_matches_delta_shape():
    # MC and delta should produce qualitatively similar ordering across two cases
    df = pd.DataFrame({
        "compute_time_q25_q75_mean": [4.0, 5.0],
        "compute_time_q25_q75_std": [0.6, 0.6],
        "epoch_time_q25_q75_mean": [10.0, 10.0],
        "epoch_time_q25_q75_std": [0.8, 0.8],
    })
    r_delta = add_compute_time_frac_epoch_quantiles(df, method="delta")
    r_mc = add_compute_time_frac_epoch_quantiles(df, method="mc", samples=5000, random_state=0)
    # Higher compute mean should lead to higher fraction quantiles for both methods
    assert r_delta.loc[1, "compute_time_frac_epoch_q25"] > r_delta.loc[0, "compute_time_frac_epoch_q25"]
    assert r_mc.loc[1, "compute_time_frac_epoch_q25"] > r_mc.loc[0, "compute_time_frac_epoch_q25"]
    assert r_delta.loc[1, "compute_time_frac_epoch_q75"] > r_delta.loc[0, "compute_time_frac_epoch_q75"]
    assert r_mc.loc[1, "compute_time_frac_epoch_q75"] > r_mc.loc[0, "compute_time_frac_epoch_q75"]


def test_handles_missing_and_zeros():
    # Should not error; should produce finite numbers where possible and NaN otherwise
    df = pd.DataFrame({
        "compute_time_q25_q75_mean": [np.nan, 1.0, 0.0],
        "compute_time_q25_q75_std": [0.1, np.nan, 0.1],
        "epoch_time_q25_q75_mean": [2.0, 2.0, 0.0],
        "epoch_time_q25_q75_std": [0.2, 0.2, 0.0],
    })
    r = add_compute_time_frac_epoch_quantiles(df, method="delta")
    assert "compute_time_frac_epoch_q25" in r.columns
    assert "compute_time_frac_epoch_q75" in r.columns

