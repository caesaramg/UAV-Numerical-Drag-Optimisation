"""
main.py
-------
UAV Aerodynamic Analysis — Drag Optimisation for Low-Altitude Surveillance

This script runs a complete aerodynamic analysis of a fixed-wing surveillance UAV:

  1. Define UAV geometry (wing, fuselage, atmosphere, mission)
  2. Solve Prandtl's Lifting Line Theory for lift and induced drag
  3. Apply component drag buildup (Raymer method) for parasite drag
  4. Generate drag polar and L/D curve
  5. Optimise wing aspect ratio and taper ratio for minimum drag

Run:
    python main.py

Outputs (saved to ./results/):
    lift_distribution.png
    drag_polar.png
    drag_breakdown.png
    LD_ratio.png
    AR_optimisation.png
    taper_optimisation.png
"""

import os
import numpy as np

from geometry       import UAV, Wing, Fuselage, Atmosphere
from lifting_line   import solve_lifting_line, sweep_alpha
from drag_analysis  import compute_drag, sweep_drag
from optimisation   import optimise_aspect_ratio, optimise_taper_ratio
from visualisation  import (
    plot_lift_distribution,
    plot_drag_polar,
    plot_drag_breakdown,
    plot_LD_ratio,
    plot_AR_optimisation,
    plot_taper_optimisation,
)

# Output directory
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def out(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


# UAV CONFIGURATION

print("\n" + "═"*55)
print("  UAV CFD & DRAG OPTIMISATION ANALYSIS")
print("  Aerodynamics — Lifting Line Theory + Component Buildup")
print("═"*55)

uav = UAV(
    wing = Wing(
        span         = 1.80,     # m      — compact for portability
        root_chord   = 0.28,     # m
        taper_ratio  = 0.60,     # —      — reduces induced drag vs rectangular
        sweep_angle  = 0.0,      # deg    — unswept for simplicity at low Mach
        dihedral     = 3.0,      # deg    — lateral stability
        airfoil      = 'NACA2412',
        alpha_zero_lift = -2.1,  # deg    — NACA 2412 (cambered)
    ),
    fuselage = Fuselage(
        length   = 1.20,    # m
        diameter = 0.12,    # m  →  fineness ratio 10
    ),
    atmosphere   = Atmosphere(altitude=150.0),  # m AGL
    cruise_speed = 22.0,    # m/s  (~79 km/h) — typical EDF/pusher UAV
    mass         = 2.50,    # kg   — all-up weight
)

print(uav)
print(f"\n  Cruise CL required    : {uav.CL_cruise:.4f}")
print(f"  Wing Reynolds number  : {uav.reynolds_number_wing:.3e}")



# CRUISE POINT — Lifting Line Theory

print("\n── Solving LLT at cruise conditions ──")

# Find angle of attack that gives required CL for level flight
from optimisation import _find_alpha_for_CL
alpha_cruise = _find_alpha_for_CL(uav.wing, uav.CL_cruise, N=80)
print(f"  Cruise AoA            : {alpha_cruise:.3f}°")

llt_cruise = solve_lifting_line(uav.wing, alpha_cruise, N=80)
drag_cruise = compute_drag(uav, alpha_cruise, llt_cruise['CL'], llt_cruise['CDi'])

print(drag_cruise.summary())

# Save cruise lift distribution plot
plot_lift_distribution(llt_cruise, uav.wing,
                       save_path=out('lift_distribution.png'))
print("  ✔ Saved: lift_distribution.png")


# ALPHA SWEEP — Drag Polar + Breakdown

print("\n── Alpha sweep: -2° to 12° ──")

alpha_range = np.linspace(-2, 12, 80)
llt_sweep   = sweep_alpha(uav.wing, alpha_range, N=80)
drag_sweep  = sweep_drag(uav, alpha_range, llt_sweep)

# Cruise operating point for polar
cruise_pt = {'CL': drag_cruise.CL, 'CD': drag_cruise.CD_total}

plot_drag_polar(drag_sweep, cruise_point=cruise_pt,
                save_path=out('drag_polar.png'))
print("  ✔ Saved: drag_polar.png")

plot_drag_breakdown(drag_sweep, save_path=out('drag_breakdown.png'))
print("  ✔ Saved: drag_breakdown.png")

plot_LD_ratio(drag_sweep, save_path=out('LD_ratio.png'))
print("  ✔ Saved: LD_ratio.png")

# Print cruise L/D
idx_LD   = np.argmax(drag_sweep['LD'])
best_LD  = drag_sweep['LD'][idx_LD]
best_alpha = alpha_range[idx_LD]
print(f"\n  Best L/D = {best_LD:.2f}  at  α = {best_alpha:.1f}°")
print(f"  Cruise L/D = {drag_cruise.LD_ratio:.2f}  at  α = {alpha_cruise:.2f}°")

# ASPECT RATIO OPTIMISATION

print("\n── Aspect ratio optimisation (AR = 4 … 18) ──")

AR_opt = optimise_aspect_ratio(uav, AR_range=np.linspace(4, 18, 60), N_llt=60)

print(f"  Optimal AR            : {AR_opt['AR_optimal']:.2f}")
print(f"  Minimum CD_total      : {AR_opt['CD_optimal']:.5f}")
print(f"  Baseline AR           : {uav.wing.aspect_ratio:.2f}")

plot_AR_optimisation(AR_opt, save_path=out('AR_optimisation.png'))
print("  ✔ Saved: AR_optimisation.png")


# TAPER RATIO OPTIMISATION


print("\n── Taper ratio optimisation (λ = 0.2 … 1.0) ──")

taper_opt = optimise_taper_ratio(uav, N_llt=60)

print(f"  Optimal taper ratio   : {taper_opt['lambda_optimal']:.2f}")
print(f"  Peak span efficiency  : {taper_opt['e_optimal']:.4f}")
print(f"  Baseline λ            : {uav.wing.taper_ratio:.2f}")

plot_taper_optimisation(taper_opt, save_path=out('taper_optimisation.png'))
print("  ✔ Saved: taper_optimisation.png")


# SUMMARY

print("\n" + "═"*55)
print("  ANALYSIS COMPLETE")
print(f"  All figures saved to: ./{RESULTS_DIR}/")
print("═"*55 + "\n")
