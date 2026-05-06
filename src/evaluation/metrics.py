"""Evaluation metrics for the graph-free dual-branch forecaster.

Computes all 11 metrics: MAE, RMSE, MAPE, sMAPE, PearsonR, SpearmanR,
OutbreakAUROC, OutbreakAUPRC, CRPS, Coverage50, Coverage90.

Public metric functions:
- compute_all_metrics: returns dict of all 11 metrics
- compute_per_horizon_metrics: returns per-horizon + aggregate metrics
- compute_ood_calibration_profile: returns OOD-only calibration/sharpness
  summaries for temporal/spatial shift analysis
"""

from __future__ import annotations

import logging
import math
from typing import Any, Collection

import numpy as np
import torch
from torch import Tensor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

REPORT_METRICS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "PearsonR",
    "SpearmanR",
    "OutbreakAUROC",
    "OutbreakAUPRC",
)
PER_HORIZON_REPORT_METRICS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MAPE",
    "sMAPE",
    "PearsonR",
    "SpearmanR",
    "OutbreakAUROC",
    "OutbreakAUPRC",
)
ALL_METRICS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MAPE",
    "sMAPE",
    "PearsonR",
    "SpearmanR",
    "OutbreakAUROC",
    "OutbreakAUPRC",
    "CRPS",
    "Coverage50",
    "Coverage90",
)

# Mathematical constants
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

# Quantile constants for coverage intervals
_Z_50 = 0.6745   # scipy.stats.norm.ppf(0.75)
_Z_90 = 1.6449   # scipy.stats.norm.ppf(0.95)

# Forecast horizons: indices 0, 1, 2 correspond to horizons 1, 2, 4
_HORIZON_INDICES = [0, 1, 2]
_HORIZON_LABELS = ["h1", "h2", "h4"]

# ZINB empirical CRPS / Coverage sample count
_ZINB_NUM_SAMPLES = 500
_DEFAULT_COVERAGE_LEVELS: tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
)
_TARGET_QUANTILE_LABELS: tuple[str, ...] = ("q1", "q2", "q3", "q4")


def _compute_point_predictions(
    pred_mean: Tensor,
    pred_std: Tensor,
) -> Tensor:
    """Return point predictions for the point-only mainline."""
    del pred_std
    return pred_mean


def _extract_valid_metric_tensors(
    pred_mean: Tensor,
    pred_std: Tensor,
    targets: Tensor,
    mask: Tensor,
    cfg: dict,
    model_output: dict | None = None,
    *,
    population: Tensor | None = None,
    per_capita_base: float = 100_000.0,
) -> dict[str, Any] | None:
    """Prepare valid flattened tensors shared by evaluation helpers."""
    valid_idx = (mask > 0).nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return None

    targets_valid = targets[valid_idx]
    pred_mean_valid = pred_mean[valid_idx]
    pred_std_valid = pred_std[valid_idx]

    point_pred = _compute_point_predictions(
        pred_mean_valid, pred_std_valid,
    )

    y = targets_valid.reshape(-1)
    p = point_pred.reshape(-1)
    mu_flat = pred_mean_valid.reshape(-1)
    sigma_flat = pred_std_valid.reshape(-1)

    population_valid: Tensor | None = None
    if population is not None:
        population_valid = population[valid_idx].float()
        H = targets_valid.shape[1]
        scale_flat = (
            (population_valid / per_capita_base)
            .unsqueeze(1)
            .expand(-1, H)
            .reshape(-1)
        )
        y_count = y * scale_flat
        p_count = p * scale_flat
    else:
        scale_flat = None
        y_count = y
        p_count = p

    return {
        "valid_idx": valid_idx,
        "targets_valid": targets_valid,
        "pred_mean_valid": pred_mean_valid,
        "pred_std_valid": pred_std_valid,
        "point_pred": point_pred,
        "y": y,
        "p": p,
        "mu_flat": mu_flat,
        "sigma_flat": sigma_flat,
        "y_count": y_count,
        "p_count": p_count,
        "scale_flat": scale_flat,
        "valid_model_output": None,
        "num_valid_locations": int(valid_idx.numel()),
        "num_entries": int(y.numel()),
        "num_horizons": int(targets_valid.shape[1]),
    }


def _gaussian_crps(y: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Closed-form CRPS for a Gaussian distribution N(mu, sigma).

    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))

    Args:
        y: Observation values.   # [M]
        mu: Distribution means.  # [M]
        sigma: Distribution stds (must be > 0).  # [M]

    Returns:
        Per-element CRPS values.  # [M]
    """
    # Clamp sigma to avoid division by zero
    sigma = sigma.clamp(min=1e-6)  # [M]

    z = (y - mu) / sigma  # [M]

    # Standard normal CDF and PDF via torch distributions
    normal = torch.distributions.Normal(0.0, 1.0)
    phi_z = torch.exp(normal.log_prob(z))  # [M] — standard normal PDF
    cdf_z = normal.cdf(z)                  # [M] — standard normal CDF

    # CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    crps = sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * phi_z - _INV_SQRT_PI)  # [M]
    return crps


def _zinb_sample(
    pi: Tensor, r: Tensor, p: Tensor, num_samples: int,
) -> Tensor:
    """Draw samples from ZINB distribution.

    Args:
        pi: Zero-inflation probability.  # [M]
        r: Dispersion parameter.          # [M]
        p: NB success probability.        # [M]
        num_samples: Number of samples to draw.

    Returns:
        Samples tensor.  # [num_samples, M]
    """
    M = pi.shape[0]

    # Sample zero-inflation: Bernoulli(pi) → 1 means "structural zero"
    zero_mask = torch.bernoulli(pi.unsqueeze(0).expand(num_samples, M))  # [S, M]

    # Sample NB component: NB(r, p) via Gamma-Poisson mixture
    # NB(r, p): mean = r*(1-p)/p, parameterized as Gamma(r, p/(1-p)) then Poisson
    # PyTorch NegativeBinomial: total_count=r, probs=1-p (prob of "failure")
    nb_dist = torch.distributions.NegativeBinomial(
        total_count=r.unsqueeze(0).expand(num_samples, M),
        probs=(1.0 - p).unsqueeze(0).expand(num_samples, M),
    )
    nb_samples = nb_dist.sample()  # [S, M]

    # ZINB: zero_mask=1 → 0, zero_mask=0 → NB sample
    samples = (1.0 - zero_mask) * nb_samples  # [S, M]
    return samples


def _empirical_crps_zinb(
    pi: Tensor, r: Tensor, p: Tensor, y: Tensor,
    num_samples: int = _ZINB_NUM_SAMPLES,
) -> Tensor:
    """Compute empirical CRPS for ZINB distribution.

    CRPS = E|Y - y| - 0.5 * E|Y - Y'| where Y, Y' ~ ZINB

    Args:
        pi, r, p: ZINB parameters.  # [M]
        y: Observations.             # [M]
        num_samples: Number of MC samples.

    Returns:
        Per-element CRPS.  # [M]
    """
    samples = _zinb_sample(pi, r, p, num_samples)  # [S, M]

    # E|Y - y|
    abs_diff = (samples - y.unsqueeze(0)).abs()  # [S, M]
    term1 = abs_diff.mean(dim=0)  # [M]

    # E|Y - Y'|: use two independent sample sets
    samples2 = _zinb_sample(pi, r, p, num_samples)  # [S, M]
    term2 = (samples - samples2).abs().mean(dim=0)  # [M]

    crps = term1 - 0.5 * term2  # [M]
    return crps


def _zinb_quantile(
    pi: Tensor, r: Tensor, p: Tensor, level: float,
    num_samples: int = _ZINB_NUM_SAMPLES,
) -> tuple[Tensor, Tensor]:
    """Compute empirical quantiles for ZINB coverage.

    Args:
        pi, r, p: ZINB parameters.  # [M]
        level: Coverage level (0.5 or 0.9).
        num_samples: Number of MC samples.

    Returns:
        (lower, upper) quantile tensors.  # [M], [M]
    """
    samples = _zinb_sample(pi, r, p, num_samples)  # [S, M]

    alpha = (1.0 - level) / 2.0
    lower = torch.quantile(samples, alpha, dim=0)        # [M]
    upper = torch.quantile(samples, 1.0 - alpha, dim=0)  # [M]
    return lower, upper


def _compute_interval_bounds(
    mu: Tensor,
    sigma: Tensor,
    level: float,
    distribution: str,
    model_output: dict | None = None,
    *,
    scale_flat: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Return interval bounds and the mask of entries supporting them."""
    if not (0.0 < level < 1.0):
        raise ValueError(f"Coverage level must be in (0, 1), got {level}.")

    if distribution == "point":
        return None

    if distribution == "zinb":
        if model_output is None:
            return None
        pi = model_output["zinb_pi"].reshape(-1)
        r = model_output["zinb_r"].reshape(-1)
        p = model_output["zinb_p"].reshape(-1)
        lower, upper = _zinb_quantile(pi, r, p, level)
        valid_mask = torch.ones_like(lower, dtype=torch.bool)
    else:
        valid_mask = sigma > 1e-4
        if valid_mask.sum().item() == 0:
            return None

        mu_v = mu[valid_mask]
        sigma_v = sigma[valid_mask]
        normal = torch.distributions.Normal(
            torch.tensor(0.0, device=mu.device, dtype=mu.dtype),
            torch.tensor(1.0, device=mu.device, dtype=mu.dtype),
        )
        alpha = (1.0 + level) / 2.0
        z = normal.icdf(torch.tensor(alpha, device=mu.device, dtype=mu.dtype))
        if distribution == "normal":
            lower = mu_v - z * sigma_v
            upper = mu_v + z * sigma_v
        elif distribution == "lognormal":
            lower = torch.exp(mu_v - z * sigma_v)
            upper = torch.exp(mu_v + z * sigma_v)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    if scale_flat is not None:
        scale_v = scale_flat[valid_mask]
        lower = lower * scale_v
        upper = upper * scale_v

    return lower, upper, valid_mask


def _compute_coverage(
    y: Tensor,
    mu: Tensor,
    sigma: Tensor,
    level: float,
    distribution: str,
    model_output: dict | None = None,
    *,
    scale_flat: Tensor | None = None,
) -> float:
    """Compute prediction interval coverage at a given level.

    Args:
        y: Ground truth values in RATE space when scale_flat is provided,
           otherwise raw count / natural space.    # [M]
        mu: Distribution mean params.               # [M]
        sigma: Distribution std params.             # [M]
        level: Coverage level (0.5 or 0.9).
        distribution: "normal", "lognormal", "zinb", or "point".
        model_output: Full model output dict (needed for ZINB params).
        scale_flat: Optional per-entry rate→count multiplier (shape [M]).
           When provided, interval bounds and y are multiplied by this
           scale so coverage is computed in count space.

    Returns:
        Coverage fraction in [0, 1], or nan if all entries excluded.
    """
    bounds = _compute_interval_bounds(
        mu, sigma, level, distribution, model_output, scale_flat=scale_flat,
    )
    if bounds is None:
        if distribution != "point":
            logger.warning(
                "Prediction interval unavailable for distribution=%s at level=%.2f.",
                distribution, level,
            )
        return float("nan")
    lower, upper, valid_mask = bounds
    y_v = y[valid_mask]
    if scale_flat is not None:
        y_v = y_v * scale_flat[valid_mask]

    in_interval = ((y_v >= lower) & (y_v <= upper)).float()  # [M']
    coverage = in_interval.mean().item()  # scalar
    return coverage


def _empty_ood_profile(reason: str) -> dict[str, Any]:
    """Return a schema-stable non-applicable OOD calibration profile."""
    return {
        "applicable": False,
        "distribution": "point",
        "reason": reason,
        "nominal_levels": [],
        "overall": None,
        "per_horizon": {},
        "by_target_quantile": {},
        "metadata": {},
    }


def _coverage_curve_from_subset(
    y: Tensor,
    mu: Tensor,
    sigma: Tensor,
    distribution: str,
    levels: Collection[float],
    *,
    model_output: dict | None = None,
    scale_flat: Tensor | None = None,
) -> dict[str, Any]:
    """Compute empirical coverage over multiple nominal levels."""
    nominal_levels = [float(level) for level in levels]
    empirical: list[float] = []
    gaps: list[float] = []
    valid_levels: list[float] = []

    for level in nominal_levels:
        coverage = _compute_coverage(
            y, mu, sigma, level, distribution, model_output, scale_flat=scale_flat,
        )
        empirical.append(float(coverage))
        if math.isnan(coverage):
            gaps.append(float("nan"))
        else:
            gap = abs(float(coverage) - float(level))
            gaps.append(gap)
            valid_levels.append(gap)

    mean_abs_gap = float(np.mean(valid_levels)) if valid_levels else float("nan")
    max_abs_gap = float(np.max(valid_levels)) if valid_levels else float("nan")
    return {
        "nominal_levels": nominal_levels,
        "empirical_coverage": empirical,
        "abs_gap": gaps,
        "mean_abs_gap": mean_abs_gap,
        "max_abs_gap": max_abs_gap,
    }


def _sharpness_from_subset(
    mu: Tensor,
    sigma: Tensor,
    distribution: str,
    *,
    model_output: dict | None = None,
    scale_flat: Tensor | None = None,
) -> dict[str, float]:
    """Compute interval-width sharpness summaries for supported distributions."""
    results: dict[str, float] = {}
    for level in (0.5, 0.9):
        bounds = _compute_interval_bounds(
            mu, sigma, level, distribution, model_output, scale_flat=scale_flat,
        )
        key_suffix = int(level * 100)
        if bounds is None:
            results[f"mean_interval_width_{key_suffix}"] = float("nan")
            results[f"median_interval_width_{key_suffix}"] = float("nan")
            continue

        lower, upper, _ = bounds
        widths = upper - lower
        results[f"mean_interval_width_{key_suffix}"] = widths.mean().item()
        results[f"median_interval_width_{key_suffix}"] = widths.median().item()
    return results


def _subset_model_output(
    valid_model_output: dict[str, Tensor] | None,
    subset_idx: Tensor,
) -> dict[str, Tensor] | None:
    if valid_model_output is None:
        return None
    return {name: tensor.reshape(-1)[subset_idx] for name, tensor in valid_model_output.items()}


def compute_ood_calibration_profile(
    pred_mean: Tensor,
    pred_std: Tensor,
    targets: Tensor,
    mask: Tensor,
    cfg: dict,
    model_output: dict | None = None,
    *,
    population: Tensor | None = None,
    per_capita_base: float = 100_000.0,
    coverage_levels: Collection[float] = _DEFAULT_COVERAGE_LEVELS,
    num_target_quantiles: int = 4,
) -> dict[str, Any]:
    """Build OOD-only calibration/sharpness summaries for shift analysis."""
    prepared = _extract_valid_metric_tensors(
        pred_mean, pred_std, targets, mask, cfg, model_output,
        population=population, per_capita_base=per_capita_base,
    )
    if prepared is None:
        profile = _empty_ood_profile("no_valid_locations")
        profile["metadata"] = {
            "num_valid_locations": 0,
            "num_entries": 0,
            "num_horizons": int(targets.shape[1]) if targets.ndim == 2 else 0,
        }
        return profile
    profile = _empty_ood_profile(
        "probabilistic calibration is not applicable for the point-only mainline",
    )
    profile["metadata"] = {
        "num_valid_locations": prepared["num_valid_locations"],
        "num_entries": prepared["num_entries"],
        "num_horizons": prepared["num_horizons"],
    }
    return profile


def compute_all_metrics(
    pred_mean: Tensor,
    pred_std: Tensor,
    targets: Tensor,
    mask: Tensor,
    cfg: dict,
    outbreak_threshold: float,
    model_output: dict | None = None,
    *,
    population: Tensor | None = None,
    per_capita_base: float = 100_000.0,
    metric_names: Collection[str] | None = None,
) -> dict[str, float]:
    """Compute all 11 evaluation metrics.

    Args:
        pred_mean: Predicted distribution mean.  # [N_l, H]
        pred_std: Predicted distribution std.     # [N_l, H]
        targets: Ground truth incidence (counts OR rates per
            per_capita_base if per-capita normalization is enabled).
            # [N_l, H]
        mask: Valid location mask (1=valid, 0=invalid).  # [N_l]
        cfg: Configuration dict with cfg["model"]["distribution"].
        outbreak_threshold: Threshold for outbreak detection (from training
            data). In rate space if per-capita normalization is enabled.
        model_output: Full model output dict (needed for ZINB metrics).
        population: Optional per-location population tensor [N_l]. When
            provided (non-None), MAE/RMSE/MAPE/sMAPE/Pearson/Spearman and
            Coverage are converted to count space via
            count = rate * (population / per_capita_base). Outbreak and
            CRPS stay in their natural (rate) space.
        per_capita_base: Base population used during normalization.

    Returns:
        Dict with all 11 metric names mapped to float values.
    """
    wanted = set(metric_names) if metric_names is not None else set(ALL_METRICS)

    # --- Filter to valid locations ---
    valid_idx = (mask > 0).nonzero(as_tuple=True)[0]  # [N_valid]

    if valid_idx.numel() == 0:
        logger.warning("No valid locations in mask. Returning nan for all metrics.")
        return {k: float("nan") for k in ALL_METRICS if k in wanted}

    # Extract valid entries
    targets_valid = targets[valid_idx]    # [N_valid, H]
    pred_mean_valid = pred_mean[valid_idx]  # [N_valid, H]
    pred_std_valid = pred_std[valid_idx]    # [N_valid, H]

    # Per-capita inverse transform scaffolding. When `population` is provided,
    # `scale_per_loc[i] = population[i] / per_capita_base` converts rates at
    # location i back to counts. Expanded to [M] below to align with flattened
    # (N_valid * H) tensors used by point metrics.
    if population is not None:
        population_valid = population[valid_idx].float()  # [N_valid]
    else:
        population_valid = None

    # --- Point predictions ---
    point_pred = _compute_point_predictions(
        pred_mean_valid, pred_std_valid
    )  # [N_valid, H]

    # Flatten to 1D for metric computation
    y = targets_valid.reshape(-1)     # [M] where M = N_valid * H
    p = point_pred.reshape(-1)        # [M]
    # Per-entry scale factor for rate -> count conversion (or None).
    if population_valid is not None:
        H = targets_valid.shape[1]
        scale_flat = (
            (population_valid / per_capita_base)
            .unsqueeze(1)
            .expand(-1, H)
            .reshape(-1)
        )  # [M]
        y_count = y * scale_flat
        p_count = p * scale_flat
    else:
        scale_flat = None
        y_count = y
        p_count = p

    logger.debug(
        "Computing metrics: %d valid locations, %d total entries",
        valid_idx.numel(), y.numel(),
    )

    metrics: dict[str, float] = {}

    # ========== Point Prediction Metrics (count space when applicable) ==========

    # MAE
    if "MAE" in wanted:
        metrics["MAE"] = (p_count - y_count).abs().mean().item()

    # RMSE
    if "RMSE" in wanted:
        metrics["RMSE"] = ((p_count - y_count).pow(2).mean()).sqrt().item()

    # MAPE: mean(|p - y| / max(|y|, 1.0)) * 100
    if "MAPE" in wanted:
        if (y_count == 0).all():
            metrics["MAPE"] = float("nan")
        else:
            denom = y_count.abs().clamp(min=1.0)  # [M]
            metrics["MAPE"] = ((p_count - y_count).abs() / denom).mean().item() * 100.0

    # sMAPE: mean(2 * |p - y| / (|p| + |y| + 1.0)) * 100
    if "sMAPE" in wanted:
        smape_denom = p_count.abs() + y_count.abs() + 1.0  # [M]
        metrics["sMAPE"] = (
            2.0 * (p_count - y_count).abs() / smape_denom
        ).mean().item() * 100.0

    # ========== Correlation Metrics (count space when applicable) ==========

    M = y_count.numel()  # total number of valid entries

    # PearsonR
    if "PearsonR" in wanted:
        if M < 2:
            metrics["PearsonR"] = float("nan")
        else:
            p_std = p_count.std(correction=1)  # [scalar] Bessel-corrected
            y_std = y_count.std(correction=1)  # [scalar]
            if p_std.item() == 0.0 or y_std.item() == 0.0:
                metrics["PearsonR"] = float("nan")
            else:
                # cov(p, y) / (std(p) * std(y))
                p_centered = p_count - p_count.mean()  # [M]
                y_centered = y_count - y_count.mean()  # [M]
                cov_py = (p_centered * y_centered).mean()  # scalar
                pearson_r = cov_py / (p_std * y_std)  # scalar
                metrics["PearsonR"] = pearson_r.item()

    # SpearmanR — use scipy (handles ties with average method)
    if "SpearmanR" in wanted:
        p_np = p_count.detach().cpu().numpy()
        y_np = y_count.detach().cpu().numpy()

        if np.std(p_np) == 0.0 or np.std(y_np) == 0.0:
            metrics["SpearmanR"] = float("nan")
        else:
            spearman_result = spearmanr(p_np, y_np)
            spearman_val = getattr(spearman_result, "statistic", spearman_result.correlation)
            if np.isnan(spearman_val):
                metrics["SpearmanR"] = float("nan")
            else:
                metrics["SpearmanR"] = float(spearman_val)

    # ========== Outbreak Detection Metrics (rate space) ==========

    if "OutbreakAUROC" in wanted or "OutbreakAUPRC" in wanted:
        labels = (y >= outbreak_threshold).long()  # [M] binary
        scores = p  # [M]

        labels_np = labels.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        # Check if all labels are the same class
        unique_labels = np.unique(labels_np)
        if len(unique_labels) < 2:
            logger.debug(
                "All outbreak labels are the same (%d). AUROC/AUPRC returning nan.",
                unique_labels[0],
            )
            if "OutbreakAUROC" in wanted:
                metrics["OutbreakAUROC"] = float("nan")
            if "OutbreakAUPRC" in wanted:
                metrics["OutbreakAUPRC"] = float("nan")
        else:
            if "OutbreakAUROC" in wanted:
                metrics["OutbreakAUROC"] = float(roc_auc_score(labels_np, scores_np))
            if "OutbreakAUPRC" in wanted:
                metrics["OutbreakAUPRC"] = float(average_precision_score(labels_np, scores_np))

    # ========== Probabilistic Metrics ==========

    if any(name in wanted for name in ("CRPS", "Coverage50", "Coverage90")):
        if "CRPS" in wanted:
            metrics["CRPS"] = (p - y).abs().mean().item()
        if "Coverage50" in wanted:
            metrics["Coverage50"] = float("nan")
        if "Coverage90" in wanted:
            metrics["Coverage90"] = float("nan")

    logger.debug("Metrics computed: %s", metrics)
    return metrics


def compute_gap_metrics(
    ood: dict[str, float],
    iid: dict[str, float],
    metric_names: Collection[str] = REPORT_METRICS,
) -> dict[str, float]:
    """Compute OOD-vs-IID absolute and relative gaps for report metrics.

    For each metric name present in both ``ood`` and ``iid``, returns::

        {f"{name}_gap":     ood[name] - iid[name],
         f"{name}_rel_gap": (ood[name] - iid[name]) / (|iid[name]| + 1e-12)}

    Sign convention (caller responsibility for interpretation):
      - Higher-is-worse metrics (RMSE, MAE, MAPE, sMAPE, CRPS): a positive
        gap means OOD is worse than IID.
      - Higher-is-better metrics (PearsonR, SpearmanR, OutbreakAUROC,
        OutbreakAUPRC, Coverage*): a negative gap means OOD is worse
        than IID.

    Metrics missing from either dict, or whose values are NaN, emit NaN
    placeholders so the output schema is predictable.
    """
    out: dict[str, float] = {}
    for name in metric_names:
        if name not in ood or name not in iid:
            out[f"{name}_gap"] = float("nan")
            out[f"{name}_rel_gap"] = float("nan")
            continue
        o_v = float(ood[name])
        i_v = float(iid[name])
        if math.isnan(o_v) or math.isnan(i_v):
            out[f"{name}_gap"] = float("nan")
            out[f"{name}_rel_gap"] = float("nan")
            continue
        gap = o_v - i_v
        rel = gap / (abs(i_v) + 1e-12)
        out[f"{name}_gap"] = gap
        out[f"{name}_rel_gap"] = rel
    return out


def compute_per_horizon_metrics(
    pred_mean: Tensor,
    pred_std: Tensor,
    targets: Tensor,
    mask: Tensor,
    cfg: dict,
    outbreak_threshold: float,
    model_output: dict | None = None,
    *,
    population: Tensor | None = None,
    per_capita_base: float = 100_000.0,
    metric_names: Collection[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all 11 metrics per forecast horizon and aggregated.

    Horizons: h1 (index 0), h2 (index 1), h4 (index 2).

    Args:
        pred_mean: Predicted distribution mean.  # [N_l, H]
        pred_std: Predicted distribution std.     # [N_l, H]
        targets: Ground truth incidence counts.   # [N_l, H]
        mask: Valid location mask.                # [N_l]
        cfg: Configuration dict.
        outbreak_threshold: Threshold for outbreak detection.
        model_output: Full model output dict (needed for ZINB metrics).

    Returns:
        Dict with keys "h1", "h2", "h4", "all", each mapping to
        a dict of all 11 metrics.
    """
    results: dict[str, dict[str, float]] = {}

    # Per-horizon metrics: slice along horizon dimension
    for h_idx, h_label in zip(_HORIZON_INDICES, _HORIZON_LABELS):
        if h_idx >= pred_mean.shape[1]:
            logger.warning(
                "Horizon index %d exceeds available horizons (%d). Skipping %s.",
                h_idx, pred_mean.shape[1], h_label,
            )
            continue

        # Slice single horizon: [N_l, 1]
        pm_h = pred_mean[:, h_idx : h_idx + 1]  # [N_l, 1]
        ps_h = pred_std[:, h_idx : h_idx + 1]    # [N_l, 1]
        t_h = targets[:, h_idx : h_idx + 1]      # [N_l, 1]

        results[h_label] = compute_all_metrics(
            pm_h, ps_h, t_h, mask, cfg, outbreak_threshold, None,
            population=population, per_capita_base=per_capita_base,
            metric_names=metric_names,
        )

    # Aggregate over all horizons
    results["all"] = compute_all_metrics(
        pred_mean, pred_std, targets, mask, cfg, outbreak_threshold, model_output,
        population=population, per_capita_base=per_capita_base,
        metric_names=metric_names,
    )

    return results
