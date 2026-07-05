"""
WTC Progressive Collapse Sensitivity Ensemble
==============================================
Implements the Bažant-Zhou / Schneider discrete floor model with:
  - full parameter uncertainty via Monte Carlo
  - floor-by-floor state tracking
  - arrest probability curves
  - Sobol-style sensitivity ranking (via correlation analysis)

Model: Schneider (2017) generalization of Bažant & Zhou (2002).
  Each floor impact is inelastic (momentum conservation).
  Energy balance per floor determines arrest vs. continuation.
  Parameters drawn from documented uncertainty ranges.

Physical sign convention:
  - downward positive
  - v = velocity of falling block just before impact
  - u = velocity just after impact (momentum conserved)
  - E_abs = energy dissipated by one story's columns
  - alpha = fraction of free-fall KE at initiation (accounts for partial
            resistance in the first failed story)
  - lambda_ = compaction ratio (crushed story height / original)
  - shed_frac = fraction of floor mass shed laterally per story
"""

import numpy as np
from scipy import stats
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────
# 1.  Parameter definitions and uncertainty ranges
# ─────────────────────────────────────────────

@dataclass
class CollapseParams:
    """
    All uncertain inputs to the discrete floor model.
    Documented sources given in comments.
    """
    # Upper block mass (kg)
    # Bažant & Zhou (2002): 58e6 kg (corrected downward by Schneider & Szuladziński)
    # Schneider (2017) / Urich (2007): ~30–33e6 kg
    M0: float = 33e6           # nominal: 33×10^6 kg

    # Floor (story) mass (kg)
    # WTC1: ~500,000 t total / 110 floors ≈ 4.5×10^6 kg/floor
    # but upper 15 floors much lighter; Schneider uses m/M ~ 0.077
    m_floor: float = 2.54e6   # nominal (gives m/M ~ 0.077 for M=33e6)

    # Story height (m)
    h: float = 3.8             # WTC nominal floor-to-floor

    # Number of stories in upper block (sets free-fall initiation height)
    n_upper: int = 13          # floors 98–110 of WTC1

    # Number of intact lower stories to traverse
    n_lower: int = 97          # floors 1–97 below initiation

    # Effective energy dissipation per story (MJ)
    # Bažant & Zhou (2002): 500 MJ (plastic hinge, upper bound neglecting fracture)
    # Korol & Sivakumaran (2014): empirical correction ×3–4 → 1500–2000 MJ max
    # Bažant & Le (2016): demands KoSi rescaled ×2/3 → 1000–1300 MJ
    # Schneider (2017): actual observed ~250 MJ first 4.6 s, up to 2000 MJ later
    E_abs_MJ: float = 500.0    # MJ; this is THE key uncertain parameter

    # Alpha: fraction of free-fall KE at initiation
    # = 1.0 if first story offered zero resistance (pure free-fall)
    # Observed roofline deceleration ~0.52g → alpha ~ 0.52 (Chandler 2010)
    # Bažant & Le (2011): demands alpha >= 0.794
    alpha: float = 1.0

    # Lambda: compaction ratio (crushed depth / original story height)
    # Bažant & Verdure (2007): 0.18; Schneider: 0.15
    lambda_: float = 0.18

    # Shedding fraction: fraction of each accreted floor mass shed laterally
    # per story of progression. Poorly constrained; video evidence suggests
    # substantial ejection. Range 0.0–0.5 in literature discussion.
    shed_frac: float = 0.0

    # Story-to-story resistance heterogeneity (coefficient of variation)
    # Lower floors have heavier columns; upper floors lighter.
    # We model E_abs as varying floor-by-floor with this CV around E_abs_MJ.
    E_abs_cv: float = 0.0      # 0 = uniform; 0.3 = 30% std dev

    # Column capacity taper factor: ratio of base-floor to initiation-zone
    # dissipation capacity, applied as a linear ramp across n_lower floors.
    # Rough estimate from NIST column schedule data (heavier sections lower
    # in the tower). Active throughout the main ensemble (Section 4).
    # Set to 1.0 to disable tapering -- e.g. to validate the closed-form
    # Schneider (2017) formula, which assumes spatially uniform capacity.
    taper_factor: float = 2.5

def make_param_distributions(n_samples: int, seed: int = 42) -> list[CollapseParams]:
    """
    Draw n_samples parameter sets from documented uncertainty ranges.
    All distributions are uniform over ranges derived from the literature.
    """
    rng = np.random.default_rng(seed)

    # Uncertainty ranges (min, max) — all from cited literature
    ranges = {
        # M0: 30–58×10^6 kg (Schneider/Urich lower, Bažant original upper)
        "M0":         (30e6,  58e6),
        # m_floor: implied by m/M ratio 0.063–0.10
        # with M0 varying we sample m/M ratio instead and compute m_floor
        # m/M in [0.063, 0.10]
        "m_over_M":   (0.063, 0.10),
        # E_abs: 250–2000 MJ (spanning observed low and KoSi empirical max)
        "E_abs_MJ":   (250.0, 2000.0),
        # alpha: 0.52–1.0 (observed roofline to pure free-fall)
        "alpha":      (0.52,  1.0),
        # lambda_: 0.15–0.22
        "lambda_":    (0.15,  0.22),
        # shed_frac: 0.0–0.50 (poorly constrained)
        "shed_frac":  (0.0,   0.50),
        # E_abs_cv: 0.0–0.40 (floor-to-floor variation)
        "E_abs_cv":   (0.0,   0.40),
    }

    samples = []
    M0_arr       = rng.uniform(*ranges["M0"],       n_samples)
    mM_arr       = rng.uniform(*ranges["m_over_M"], n_samples)
    Eabs_arr     = rng.uniform(*ranges["E_abs_MJ"], n_samples)
    alpha_arr    = rng.uniform(*ranges["alpha"],    n_samples)
    lambda_arr   = rng.uniform(*ranges["lambda_"],  n_samples)
    shed_arr     = rng.uniform(*ranges["shed_frac"],n_samples)
    cv_arr       = rng.uniform(*ranges["E_abs_cv"], n_samples)

    for i in range(n_samples):
        M0 = M0_arr[i]
        m_floor = M0 * mM_arr[i]
        samples.append(CollapseParams(
            M0        = M0,
            m_floor   = m_floor,
            E_abs_MJ  = Eabs_arr[i],
            alpha     = alpha_arr[i],
            lambda_   = lambda_arr[i],
            shed_frac = shed_arr[i],
            E_abs_cv  = cv_arr[i],
        ))

    return samples


# ─────────────────────────────────────────────
# 2.  Core floor-by-floor simulator
# ─────────────────────────────────────────────

@dataclass
class SimResult:
    arrested: bool           # True if collapse arrested before bottom
    arrest_floor: Optional[int]   # floor index at arrest (None if total)
    floors_crushed: int      # total floors crushed
    final_KE_MJ: float       # kinetic energy at final state (MJ)
    velocity_profile: list   # v before each impact (m/s)
    mass_profile: list       # effective falling mass at each floor (kg)
    KE_profile: list         # KE (MJ) just before each impact
    E_available: list        # available dissipation per floor (MJ, with heterogeneity)


def run_sim(p: CollapseParams, rng: Optional[np.random.Generator] = None) -> SimResult:
    """
    Simulate discrete floor-by-floor progressive collapse.

    State after crushing floor k (0-indexed from initiation point):
      - M_eff: effective mass of falling block (after shedding)
      - v: velocity just before next impact

    Returns SimResult with full state history.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    g = 9.81  # m/s²
    h_eff = p.h * (1.0 - p.lambda_)  # effective drop per story after compaction

    # Per-floor E_abs with heterogeneity
    # Lower floors (higher index) have heavier columns → scale E_abs upward
    # We use a simple linear taper: E_abs increases by factor taper_factor
    # over the full height of the lower structure. This is a model assumption.
    # We apply it multiplicatively on top of the drawn E_abs_MJ.

    # Generate per-floor dissipation capacities
    n_floors = p.n_lower
    E_base = np.array([
        p.E_abs_MJ * (1.0 + (p.taper_factor - 1.0) * k / max(n_floors - 1, 1))
        for k in range(n_floors)
    ])  # MJ, increasing toward base

    if p.E_abs_cv > 0:
        # Multiplicative lognormal noise on each floor
        sigma_log = np.sqrt(np.log(1 + p.E_abs_cv**2))
        mu_log    = -0.5 * sigma_log**2
        noise     = rng.lognormal(mu_log, sigma_log, n_floors)
        E_floor   = E_base * noise   # MJ per floor
    else:
        E_floor = E_base.copy()

    # Initial conditions: upper block falls through failed story
    # KE after falling h_eff with fraction alpha of free-fall
    # (alpha<1 means some energy was absorbed in the first failed story)
    M_eff = p.M0  # effective mass accretes as floors are added
    v0_sq = 2 * p.alpha * g * h_eff
    v = np.sqrt(max(v0_sq, 0.0))   # velocity before first impact

    velocity_profile = []
    mass_profile     = []
    KE_profile       = []
    E_available      = list(E_floor * 1e6)  # convert to J for calculations

    arrested        = False
    arrest_floor    = None
    floors_crushed  = 0

    for k in range(n_floors):
        KE_before = 0.5 * M_eff * v**2   # J, before impact

        velocity_profile.append(v)
        mass_profile.append(M_eff)
        KE_profile.append(KE_before / 1e6)  # MJ

        # Inelastic impact: accrete floor mass (before shedding)
        m_accreted = p.m_floor
        M_new = M_eff + m_accreted
        v_after_impact = (M_eff / M_new) * v   # momentum conservation

        # Apply shedding: fraction of accreted mass lost laterally
        # Shed mass carries away some momentum; we conservatively assume
        # shed mass leaves at v_after_impact (no deceleration from shedding)
        M_shed = p.shed_frac * m_accreted
        M_eff_new = M_new - M_shed

        # Energy available to crush next story:
        # KE after impact + gravitational PE gained descending h_eff
        E_kinetic = 0.5 * M_eff_new * v_after_impact**2   # J
        E_gravity = M_eff_new * g * h_eff                  # J
        E_total_available = E_kinetic + E_gravity           # J

        E_diss = E_available[k]   # J this floor can absorb

        floors_crushed += 1

        if E_diss >= E_total_available:
            # Arrest: dissipation exceeds available energy
            arrested = True
            arrest_floor = k
            final_KE_MJ = 0.0
            break
        else:
            # Continue: compute velocity before next impact
            E_residual = E_total_available - E_diss   # J
            # E_residual = 0.5 * M_eff_new * v_next^2 (at bottom of story drop)
            v_next = np.sqrt(2 * E_residual / M_eff_new)
            M_eff = M_eff_new
            v = v_next
    else:
        # Reached bottom without arrest
        arrested = False
        arrest_floor = None
        final_KE_MJ = 0.5 * M_eff * v**2 / 1e6

    return SimResult(
        arrested        = arrested,
        arrest_floor    = arrest_floor,
        floors_crushed  = floors_crushed,
        final_KE_MJ     = final_KE_MJ,
        velocity_profile = [float(x) for x in velocity_profile],
        mass_profile     = [float(x) for x in mass_profile],
        KE_profile       = [float(x) for x in KE_profile],
        E_available      = [float(x)/1e6 for x in E_available],  # back to MJ
    )


# ─────────────────────────────────────────────
# 3.  Ensemble runner
# ─────────────────────────────────────────────

def run_ensemble(n_samples: int = 10000, seed: int = 42):
    """Run full Monte Carlo ensemble, return results + param arrays."""
    param_sets = make_param_distributions(n_samples, seed=seed)
    rng_master = np.random.default_rng(seed + 1)

    results = []
    for i, p in enumerate(param_sets):
        rng_i = np.random.default_rng(rng_master.integers(0, 2**32))
        r = run_sim(p, rng=rng_i)
        results.append(r)

    return param_sets, results


# ─────────────────────────────────────────────
# 4.  Analysis helpers
# ─────────────────────────────────────────────

def compute_arrest_probability(param_sets, results):
    """Overall arrest probability and by-floor breakdown."""
    n = len(results)
    n_arrested = sum(r.arrested for r in results)
    p_arrest = n_arrested / n

    # Floor-wise cumulative arrest probability
    # P(collapse arrested at or before floor k)
    max_floors = max(r.floors_crushed for r in results)
    arrest_by_floor = np.zeros(max_floors + 1)
    for r in results:
        if r.arrested and r.arrest_floor is not None:
            arrest_by_floor[r.arrest_floor] += 1
    cum_arrest = np.cumsum(arrest_by_floor) / n

    return p_arrest, cum_arrest


def sensitivity_analysis(param_sets, results):
    """
    Rank parameters by Pearson correlation with floors_crushed.
    A large |r| means that parameter strongly predicts outcome.
    """
    outcomes = np.array([r.floors_crushed for r in results], dtype=float)

    param_arrays = {
        "E_abs_MJ":   np.array([p.E_abs_MJ   for p in param_sets]),
        "M0 (kg)":    np.array([p.M0         for p in param_sets]),
        "alpha":      np.array([p.alpha       for p in param_sets]),
        "lambda_":    np.array([p.lambda_     for p in param_sets]),
        "shed_frac":  np.array([p.shed_frac   for p in param_sets]),
        "E_abs_cv":   np.array([p.E_abs_cv    for p in param_sets]),
        "m_floor":    np.array([p.m_floor     for p in param_sets]),
    }

    correlations = {}
    for name, arr in param_arrays.items():
        r, pval = stats.pearsonr(arr, outcomes)
        correlations[name] = (r, pval)

    # Sort by |r|
    ranked = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)
    return ranked


def threshold_sweep(E_abs_values, n_samples=2000, seed=99):
    """
    For a range of E_abs values, compute arrest probability
    holding all other params at nominal.
    """
    arrest_probs = []
    rng = np.random.default_rng(seed)
    for E in E_abs_values:
        n_arr = 0
        for _ in range(n_samples):
            p = CollapseParams(E_abs_MJ=E)
            r = run_sim(p, rng=rng)
            if r.arrested:
                n_arr += 1
        arrest_probs.append(n_arr / n_samples)
    return arrest_probs


def shedding_sweep(shed_values, n_samples=2000, seed=77):
    """Arrest probability as function of shedding fraction."""
    arrest_probs = []
    rng = np.random.default_rng(seed)
    for s in shed_values:
        n_arr = 0
        for _ in range(n_samples):
            p = CollapseParams(E_abs_MJ=750.0, shed_frac=s)
            r = run_sim(p, rng=rng)
            if r.arrested:
                n_arr += 1
        arrest_probs.append(n_arr / n_samples)
    return arrest_probs


# ─────────────────────────────────────────────
# 5.  Main: run everything, save results JSON
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running main ensemble (N=10,000)...")
    param_sets, results = run_ensemble(n_samples=10_000, seed=42)

    p_arrest, cum_arrest = compute_arrest_probability(param_sets, results)
    ranked = sensitivity_analysis(param_sets, results)

    print(f"\n{'='*55}")
    print(f"  OVERALL ARREST PROBABILITY: {p_arrest:.3f} ({p_arrest*100:.1f}%)")
    print(f"{'='*55}")

    print("\nSENSITIVITY RANKING (Pearson r with floors_crushed):")
    print(f"  {'Parameter':<18} {'r':>8}  {'p-value':>12}")
    print(f"  {'-'*42}")
    for name, (r, pv) in ranked:
        sig = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
        print(f"  {name:<18} {r:>8.4f}  {pv:>12.2e}  {sig}")

    print("\nFLOOR-WISE CUMULATIVE ARREST PROBABILITY (selected floors):")
    checkpoints = [1, 5, 10, 20, 30, 50, 70, 97]
    for cp in checkpoints:
        if cp < len(cum_arrest):
            print(f"  Floor {cp:>3}: P(arrested by here) = {cum_arrest[cp]:.4f}")

    # E_abs threshold sweep
    print("\nE_ABS THRESHOLD SWEEP (other params nominal):")
    E_vals = np.linspace(250, 2000, 36)
    ap = threshold_sweep(E_vals, n_samples=2000)
    for E, prob in zip(E_vals[::4], ap[::4]):
        bar = "█" * int(prob * 30)
        print(f"  E={E:6.0f} MJ  P(arrest)={prob:.3f}  {bar}")

    # Shedding sweep
    print("\nSHEDDING FRACTION SWEEP (E_abs=750 MJ, other params nominal):")
    s_vals = np.linspace(0, 0.5, 21)
    sp = shedding_sweep(s_vals, n_samples=2000)
    for s, prob in zip(s_vals[::2], sp[::2]):
        bar = "█" * int(prob * 30)
        print(f"  shed={s:.2f}  P(arrest)={prob:.3f}  {bar}")

    # Save compact results for plotting
    out = {
        "p_arrest_overall": p_arrest,
        "cum_arrest_by_floor": cum_arrest.tolist(),
        "sensitivity": {name: {"r": r, "pval": pv} for name, (r, pv) in ranked},
        "E_sweep": {"E_MJ": E_vals.tolist(), "p_arrest": ap},
        "shed_sweep": {"shed_frac": s_vals.tolist(), "p_arrest": sp},
        "n_samples": 10_000,
        "n_arrested": int(p_arrest * 10_000),
    }
    with open("sim_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to sim_results.json")
