"""
optimisation.py
---------------
Aerodynamic optimisation of the UAV wing geometry.

Two optimisation studies:
1. Aspect Ratio sweep  — fixed wing area, vary AR → find minimum CD_total at cruise CL
2. Taper Ratio sweep   — fixed AR, vary λ → minimise induced drag (maximise span efficiency)

The constraint is: lift = weight at cruise (CL fixed by mission).
We ask: which AR and λ minimise total drag?

Result: there is an optimal AR balancing reduced CDi against increased CD0
(higher AR → longer, thinner wing → more wetted area → more skin friction).
"""

import numpy as np
import copy
from typing import Tuple

from geometry import Wing, UAV
from lifting_line import solve_lifting_line
from drag_analysis import compute_drag

# Aspect Ratio optimisation


def optimise_aspect_ratio(
    base_uav:    UAV,
    AR_range:    np.ndarray = None,
    N_llt:       int = 60,
) -> dict:
    """
    Sweep aspect ratio at constant wing area and cruise condition.

    For each AR:
      - wing is resized (span changes, area constant, root chord adjusts)
      - LLT solved at cruise CL
      - total drag computed
    """
    if AR_range is None:
        AR_range = np.linspace(4, 18, 70)

    S_ref   = base_uav.wing.area            # keep area constant
    lambda_ = base_uav.wing.taper_ratio     # keep taper constant

    CD_total_arr = np.zeros_like(AR_range)
    CD0_arr      = np.zeros_like(AR_range)
    CDi_arr      = np.zeros_like(AR_range)
    e_arr        = np.zeros_like(AR_range)
    span_arr     = np.zeros_like(AR_range)
    CL_arr       = np.zeros_like(AR_range)
    LD_arr       = np.zeros_like(AR_range)

    for i, AR in enumerate(AR_range):
        # Derive wing dimensions for this AR at constant area
        # S = b²/AR  →  b = sqrt(AR × S)
        b  = np.sqrt(AR * S_ref)
        # For trapezoidal wing: S = b/2 × cr × (1 + λ)  →  cr = 2S/(b(1+λ))
        cr = 2 * S_ref / (b * (1 + lambda_))

        uav = copy.deepcopy(base_uav)
        uav.wing.span       = b
        uav.wing.root_chord = cr
        # (taper_ratio, sweep, etc. stay the same)

        # Cruise CL required (L = W) stays same (same weight, speed, S)
        CL_target = uav.CL_cruise

        # Find AoA that achieves CL_target via LLT
        # Simple bisection between -2° and 15°
        alpha_sol = _find_alpha_for_CL(uav.wing, CL_target, N_llt)
        if alpha_sol is None:
            CD_total_arr[i] = np.nan
            continue

        llt = solve_lifting_line(uav.wing, alpha_sol, N_llt)
        db  = compute_drag(uav, alpha_sol, llt['CL'], llt['CDi'])

        CD_total_arr[i] = db.CD_total
        CD0_arr[i]      = db.CD0
        CDi_arr[i]      = db.CDi
        e_arr[i]        = llt['span_efficiency']
        span_arr[i]     = b
        CL_arr[i]       = llt['CL']
        LD_arr[i]       = db.LD_ratio

    # Optimal AR = minimum total drag
    valid      = ~np.isnan(CD_total_arr)
    idx_opt    = np.nanargmin(CD_total_arr)
    AR_optimal = AR_range[idx_opt]

    return {
        'AR_range':   AR_range,
        'CD_total':   CD_total_arr,
        'CD0':        CD0_arr,
        'CDi':        CDi_arr,
        'e':          e_arr,
        'span':       span_arr,
        'CL':         CL_arr,
        'LD':         LD_arr,
        'AR_optimal': AR_optimal,
        'CD_optimal': CD_total_arr[idx_opt],
        'idx_opt':    idx_opt,
    }

# Taper Ratio optimisation

def optimise_taper_ratio(
    base_uav: UAV,
    lambda_range: np.ndarray = None,
    N_llt: int = 60,
) -> dict:
    """
    Sweep taper ratio at fixed AR and wing area.
    Objective: maximise span efficiency e.
    """
    if lambda_range is None:
        lambda_range = np.linspace(0.20, 1.0, 50)

    AR    = base_uav.wing.aspect_ratio
    S_ref = base_uav.wing.area

    e_arr       = np.zeros_like(lambda_range)
    CDi_arr     = np.zeros_like(lambda_range)
    CD0_arr     = np.zeros_like(lambda_range)
    CD_total_arr= np.zeros_like(lambda_range)

    CL_target = base_uav.CL_cruise

    for i, lam in enumerate(lambda_range):
        b  = np.sqrt(AR * S_ref)
        cr = 2 * S_ref / (b * (1 + lam))

        uav = copy.deepcopy(base_uav)
        uav.wing.span        = b
        uav.wing.root_chord  = cr
        uav.wing.taper_ratio = lam

        alpha_sol = _find_alpha_for_CL(uav.wing, CL_target, N_llt)
        if alpha_sol is None:
            e_arr[i] = np.nan
            continue

        llt = solve_lifting_line(uav.wing, alpha_sol, N_llt)
        db  = compute_drag(uav, alpha_sol, llt['CL'], llt['CDi'])

        e_arr[i]        = llt['span_efficiency']
        CDi_arr[i]      = llt['CDi']
        CD0_arr[i]      = db.CD0
        CD_total_arr[i] = db.CD_total

    idx_opt      = np.nanargmax(e_arr)
    lambda_opt   = lambda_range[idx_opt]

    return {
        'lambda_range':  lambda_range,
        'e':             e_arr,
        'CDi':           CDi_arr,
        'CD0':           CD0_arr,
        'CD_total':      CD_total_arr,
        'lambda_optimal':lambda_opt,
        'e_optimal':     e_arr[idx_opt],
    }


# Helper: bisection to find AoA that yields target CL


def _find_alpha_for_CL(
    wing,
    CL_target: float,
    N: int = 60,
    alpha_lo: float = -4.0,
    alpha_hi: float = 15.0,
    tol: float = 1e-5,
) -> float:
    """
    Bisection search for AoA [deg] that achieves CL_target via LLT.
    Returns None if not solvable in given bracket.
    """
    from lifting_line import solve_lifting_line as _llt
    f_lo = _llt(wing, alpha_lo, N)['CL'] - CL_target
    f_hi = _llt(wing, alpha_hi, N)['CL'] - CL_target

    if f_lo * f_hi > 0:
        return None   # root not bracketed

    for _ in range(60):
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        f_mid     = _llt(wing, alpha_mid, N)['CL'] - CL_target
        if abs(f_mid) < tol:
            return alpha_mid
        if f_lo * f_mid < 0:
            alpha_hi = alpha_mid
            f_hi     = f_mid
        else:
            alpha_lo = alpha_mid
            f_lo     = f_mid

    return 0.5 * (alpha_lo + alpha_hi)
