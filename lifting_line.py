"""
lifting_line.py
---------------
Prandtl Lifting Line Theory (LLT) solver for finite wings.

Theory:
-------
LLT models the spanwise variation of circulation Γ(y) using a Fourier series.
The wing is represented by a bound vortex filament with trailing vortex sheets.

Cosine substitution:  y = -(b/2) cos(θ),   θ ∈ [0, π]

Circulation expansion:
    Γ(θ) = 2 b V∞ Σ_{n=1}^{N} A_n sin(nθ)

The fundamental LLT equation at N control points θ_i:
    Σ_n A_n sin(nθ_i) [ 4b/(c_i a₀) + n/sin(θ_i) ] = α(θ_i) − α_{L0}

Solving this linear system gives coefficients A_n, from which:
    C_L  = π AR A₁
    C_Di = π AR Σ_n n A_n²  =  C_L² / (π AR e)
    e    = 1 / (1 + δ),   δ = Σ_{n≥2} n (A_n/A₁)²

An elliptic load distribution (A_n = 0 for n > 1) gives e = 1 (ideal).
Taper and sweep reduce e below unity.

Reference: Anderson, J.D., "Fundamentals of Aerodynamics", Ch. 5.
"""

import numpy as np
from typing import Dict, Any


def solve_lifting_line(wing, alpha_deg: float, N: int = 80) -> Dict[str, Any]:
    """
    Solve Prandtl's Lifting Line Theory for a tapered wing.

    Parameters
    ----------
    wing      : Wing object (from geometry.py)
    alpha_deg : Geometric angle of attack [degrees]
    N         : Number of Fourier terms / control points (even N → N+1)

    Returns
    -------
    dict with keys:
        CL              – 3D lift coefficient
        CDi             – induced drag coefficient
        span_efficiency – Oswald span efficiency factor e
        y               – spanwise control point positions [m]
        Gamma           – normalised circulation Γ/(2bV∞) at each point
        lift_dist       – local cl·c(y) distribution [proportional to Γ]
        A               – Fourier coefficients A_n
        alpha_deg       – angle of attack used
    """
    if N % 2 == 0:
        N += 1   # prefer odd so matrix is better conditioned

    b   = wing.span
    AR  = wing.aspect_ratio
    a0  = wing.a0                              # 2D lift curve slope [rad⁻¹]
    aL0 = np.radians(wing.alpha_zero_lift)     # zero-lift AoA [rad]
    alpha = np.radians(alpha_deg)              # geometric AoA [rad]

    # Control point angles — avoid endpoints (singularity at θ=0,π)
    theta = np.array([i * np.pi / (N + 1) for i in range(1, N + 1)])
    y     = -(b / 2) * np.cos(theta)           # spanwise positions

    # Local chord at each control point
    c = np.array([wing.chord_at(yi) for yi in y])

    # Harmonic indices n = 1, 2, …, N
    n = np.arange(1, N + 1)                   # shape (N,)

    # Build coefficient matrix  M[i, j] = sin(n_j θ_i) [4b/(c_i a0) + n_j/sin(θ_i)]
    # Broadcasting: theta (N,1), n (1,N)
    theta_col = theta[:, np.newaxis]           # (N, 1)
    n_row     = n[np.newaxis, :]               # (1, N)

    M = np.sin(n_row * theta_col) * (
        4 * b / (c[:, np.newaxis] * a0) + n_row / np.sin(theta_col)
    )                                          # shape (N, N)

    # Right-hand side: effective incidence at each control point
    rhs = (alpha - aL0) * np.ones(N)

    # Solve linear system  M A = rhs
    A = np.linalg.solve(M, rhs)               # Fourier coefficients

    # ---- Aerodynamic coefficients ----
    CL = A[0] * np.pi * AR

    # Span efficiency:  1/e = 1 + δ,  δ = Σ_{n≥2} n (A_n/A_1)²
    delta = np.sum((n[1:] + 1) * (A[1:] / A[0]) ** 2) if N > 1 else 0.0
    e     = 1.0 / (1.0 + delta)

    CDi = CL ** 2 / (np.pi * AR * e)

    # ---- Spanwise distributions ----
    # Normalised circulation: γ(θ) = Γ/(2bV∞) = Σ A_n sin(nθ)
    Gamma = np.einsum('j,ij->i', A, np.sin(np.outer(theta, n)))

    # Local section lift coefficient × chord  cl·c(y)  ∝  Γ(y)
    # cl(y) = 2Γ(y)/(V∞ c(y)),  lift_dist = cl(y)·c(y) = 2Γ(y)/V∞
    # Using normalised Γ̂ = Γ/(2bV∞)  →  lift_dist = 4b·Γ̂(y)
    lift_dist = 4 * b * Gamma

    # Elliptic reference distribution (normalised to match same CL)
    Gamma_elliptic = (CL / (np.pi * AR)) * np.sin(theta)
    lift_dist_elliptic = 4 * b * Gamma_elliptic

    return {
        'CL':               CL,
        'CDi':              CDi,
        'span_efficiency':  e,
        'y':                y,
        'theta':            theta,
        'Gamma':            Gamma,
        'lift_dist':        lift_dist,
        'lift_dist_elliptic': lift_dist_elliptic,
        'A':                A,
        'n':                n,
        'alpha_deg':        alpha_deg,
    }


def sweep_alpha(wing, alpha_range: np.ndarray, N: int = 80) -> Dict[str, np.ndarray]:
    """
    Run LLT over a range of angles of attack.

    Returns arrays of CL, CDi, e, and (CL, CDi) pairs for the drag polar.
    """
    results = [solve_lifting_line(wing, a, N) for a in alpha_range]

    return {
        'alpha':            alpha_range,
        'CL':               np.array([r['CL']  for r in results]),
        'CDi':              np.array([r['CDi'] for r in results]),
        'span_efficiency':  np.array([r['span_efficiency'] for r in results]),
    }
