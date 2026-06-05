"""
Extreme Value Theory (EVT) Module — Peak-Over-Threshold with GPD
=================================================================

This module implements the mathematical core of Phase 2's risk-aware
provisioning. It fits a Generalized Pareto Distribution (GPD) to the
tail of the forecast error distribution and computes CVaR (Conditional
Value-at-Risk / Expected Shortfall).

Methodology:
    1. Peak-Over-Threshold (POT): Select a high threshold u (e.g., P90
       of the standardized residuals). Extract exceedances x_i = z_i - u
       for all z_i > u.

    2. GPD Fit: Fit a Generalized Pareto Distribution to the exceedances
       using Maximum Likelihood Estimation. The GPD has two parameters:
           xi (shape) — controls tail heaviness
           beta (scale) — controls spread

    3. VaR: Value-at-Risk at confidence alpha is the quantile of the
       fitted tail distribution:
           VaR(alpha) = u + (beta / xi) * [((1-alpha)/P(z>u))^(-xi) - 1]

    4. CVaR: Conditional Value-at-Risk (Expected Shortfall) is the
       expected loss given that loss exceeds VaR:
           CVaR(alpha) = VaR(alpha)/(1-xi) + (beta - xi*u)/(1-xi)

    This requires xi < 1 for finite CVaR.

Design:
    This module operates on ALREADY-STANDARDIZED residuals. The caller
    is responsible for computing z_t = epsilon_t / sigma_t before
    passing data to this module. This separation keeps the math pure
    and independently testable.

References:
    - Pickands (1975), "Statistical Inference Using Extreme Order Statistics"
    - McNeil & Frey (2000), "Estimation of tail-related risk measures"
    - Embrechts, Kluppelberg & Mikosch (1997), "Modelling Extremal Events"
"""

import numpy as np
from scipy.stats import genpareto


def fit_gpd(exceedances: np.ndarray) -> tuple:
    """
    Fit a Generalized Pareto Distribution to exceedance data via MLE.

    Parameters
    ----------
    exceedances : np.ndarray
        Positive values representing (z_i - threshold) for z_i > threshold.
        Must have length >= 10 for a meaningful fit.

    Returns
    -------
    tuple (xi, beta)
        xi : float — shape parameter (tail index)
        beta : float — scale parameter

    Raises
    ------
    ValueError
        If fewer than 10 exceedances are provided.
    """
    if len(exceedances) < 10:
        raise ValueError(
            f"GPD fit requires >= 10 exceedances, got {len(exceedances)}. "
            f"Consider lowering the threshold percentile."
        )

    # scipy.stats.genpareto parameterization:
    #   genpareto.fit(data, floc=0) returns (c, loc, scale)
    #   where c = xi (shape), scale = beta
    #   floc=0 because exceedances are already shifted by the threshold
    c, _loc, scale = genpareto.fit(exceedances, floc=0)

    xi = float(c)
    beta = float(scale)

    if beta <= 0:
        raise ValueError(f"GPD scale parameter beta must be > 0, got {beta}")

    return xi, beta


def compute_var(xi: float, beta: float, threshold_u: float,
                exceedance_prob: float, alpha: float) -> float:
    """
    Compute Value-at-Risk at confidence level alpha using POT-GPD.

    VaR(alpha) = u + (beta / xi) * [((1 - alpha) / P(z > u))^(-xi) - 1]

    For xi == 0 (exponential tail), uses the limiting form:
    VaR(alpha) = u + beta * log((1 - alpha) / P(z > u))     [negated]

    Parameters
    ----------
    xi : float — GPD shape parameter
    beta : float — GPD scale parameter
    threshold_u : float — POT threshold
    exceedance_prob : float — P(z > u), the probability of exceeding threshold
    alpha : float — confidence level (e.g., 0.99)

    Returns
    -------
    float — VaR at confidence alpha
    """
    if exceedance_prob <= 0 or exceedance_prob >= 1:
        raise ValueError(f"exceedance_prob must be in (0, 1), got {exceedance_prob}")
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if abs(xi) < 1e-10:
        # Exponential tail (xi -> 0 limit)
        var = threshold_u + beta * np.log(exceedance_prob / (1 - alpha))
    else:
        var = threshold_u + (beta / xi) * (
            ((1 - alpha) / exceedance_prob) ** (-xi) - 1
        )

    return float(var)


def compute_cvar(xi: float, beta: float, var_value: float,
                 threshold_u: float) -> float:
    """
    Compute Conditional Value-at-Risk (Expected Shortfall) beyond VaR.

    CVaR(alpha) = VaR(alpha) / (1 - xi) + (beta - xi * u) / (1 - xi)

    Parameters
    ----------
    xi : float — GPD shape parameter (must be < 1 for finite CVaR)
    beta : float — GPD scale parameter
    var_value : float — VaR at the target confidence level
    threshold_u : float — POT threshold

    Returns
    -------
    float — CVaR (Expected Shortfall)

    Raises
    ------
    ValueError
        If xi >= 1 (CVaR is infinite).
    """
    if xi >= 1.0:
        raise ValueError(
            f"CVaR is infinite for xi >= 1 (heavy tail). Got xi = {xi:.4f}. "
            f"Consider raising the threshold percentile or using a lighter "
            f"tail assumption."
        )

    cvar = var_value / (1 - xi) + (beta - xi * threshold_u) / (1 - xi)
    return float(cvar)


def fit_evt_pipeline(standardized_residuals: np.ndarray,
                     threshold_percentile: float = 90,
                     alpha: float = 0.99) -> dict:
    """
    Full EVT pipeline: threshold selection -> GPD fit -> VaR -> CVaR.

    Parameters
    ----------
    standardized_residuals : np.ndarray
        Residuals already divided by their conditional volatility (z_t).
    threshold_percentile : float
        Percentile for the POT threshold (default: 90).
    alpha : float
        Confidence level for VaR/CVaR (default: 0.99).

    Returns
    -------
    dict with keys:
        threshold_u : float — the POT threshold value
        threshold_percentile : float — the percentile used
        n_total : int — total number of residuals
        n_exceedances : int — number of exceedances
        exceedance_prob : float — fraction exceeding threshold
        xi : float — GPD shape parameter
        beta : float — GPD scale parameter
        var : float — Value-at-Risk at alpha
        cvar_z : float — CVaR of the standardized distribution
        alpha : float — confidence level used
    """
    z = np.asarray(standardized_residuals, dtype=np.float64)

    # Step 1: Select threshold
    threshold_u = float(np.percentile(z, threshold_percentile))

    # Step 2: Extract exceedances
    mask = z > threshold_u
    exceedances = z[mask] - threshold_u
    n_exceedances = int(mask.sum())
    exceedance_prob = n_exceedances / len(z)

    print(f"    [EVT] Threshold (P{threshold_percentile:.0f}): {threshold_u:.4f}")
    print(f"    [EVT] Exceedances: {n_exceedances} / {len(z)} "
          f"({100 * exceedance_prob:.1f}%)")

    # Step 3: Fit GPD
    xi, beta = fit_gpd(exceedances)
    print(f"    [EVT] GPD fit: xi={xi:.4f}, beta={beta:.4f}")

    if xi >= 1.0:
        print(f"    [EVT] WARNING: xi >= 1 implies infinite CVaR. "
              f"Tail is extremely heavy.")

    # Step 4: Compute VaR
    var = compute_var(xi, beta, threshold_u, exceedance_prob, alpha)
    print(f"    [EVT] VaR({alpha}): {var:.4f}")

    # Step 5: Compute CVaR
    cvar_z = compute_cvar(xi, beta, var, threshold_u)
    print(f"    [EVT] CVaR({alpha}): {cvar_z:.4f}")

    return {
        "threshold_u": threshold_u,
        "threshold_percentile": threshold_percentile,
        "n_total": len(z),
        "n_exceedances": n_exceedances,
        "exceedance_prob": exceedance_prob,
        "xi": xi,
        "beta": beta,
        "var": var,
        "cvar_z": cvar_z,
        "alpha": alpha,
    }
