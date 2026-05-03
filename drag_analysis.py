"""
drag_analysis.py

Component drag buildup using Raymer's equivalent-skin-friction / form-factor method.

Total drag is decomposed into:
    CD_total = CD0 + CDi

where:
    CD0  = parasite drag (zero-lift drag) — wing + fuselage + interference + misc
    CDi  = induced drag (from Lifting Line Theory)

Each component follows:
    CD_component = Cf × FF × Q × (S_wet / S_ref)

    Cf   — skin friction coefficient (flat-plate approximation)
    FF   — form factor (accounts for pressure drag / thickness effects)
    Q    — interference factor
    S_wet— wetted area of component
    S_ref— reference wing area

References:
    Raymer, D.P. (2018) "Aircraft Design: A Conceptual Approach", 6th ed.
    Schlichting, H. (1979) "Boundary Layer Theory"
    Hoerner, S.F. (1965) "Fluid Dynamic Drag"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

# Skin friction models

def Cf_laminar(Re: float) -> float:
    """Blasius flat-plate laminar skin friction coefficient."""
    return 1.328 / max(Re, 1e3) ** 0.5


def Cf_turbulent(Re: float) -> float:
    """Prandtl–Schlichting turbulent flat-plate Cf (log-law form)."""
    if Re < 1e4:
        return 0.01
    return 0.455 / (np.log10(Re) ** 2.58)


def Cf_mixed(Re: float, x_tr: float = 0.15) -> float:
    """
    Mixed laminar/turbulent Cf using transition at x_tr fraction of chord.
    For Re_wing ~ 3–5 × 10⁵ (low-altitude UAV) transition is expected
    at ~15–25% chord under free-stream turbulence.
    """
    Re_tr = x_tr * Re                   # Re at transition location
    A     = 0.441 * Re_tr ** 0.6        # Prandtl correction term
    return (0.455 / np.log10(Re) ** 2.58) - A / Re

# Form factors (Raymer §12.4 / §12.5)

def FF_wing(t_c: float, sweep_deg: float, x_c_max: float = 0.30) -> float:
    """
    Form factor for a lifting surface (Raymer Eq. 12.30).

    Parameters
    
    t_c      : thickness-to-chord ratio
    sweep_deg: sweep angle at maximum-thickness line [deg]
    x_c_max  : chordwise position of max thickness (0.30 for NACA 4-series)
    """
    cos_sweep = np.cos(np.radians(sweep_deg))
    FF = (1 + 0.6 / x_c_max * t_c + 100 * t_c ** 4) * (
         1.34 * cos_sweep ** 0.28
    )
    return FF


def FF_fuselage(f: float) -> float:
    """
    Form factor for a streamlined body of revolution (Raymer Eq. 12.31).
    f = fineness ratio = length / diameter.
    """
    return 1 + 60 / f ** 3 + f / 400


@dataclass
class DragBreakdown:
    """Container for all drag contributions and derived quantities."""
    alpha_deg:        float
    CL:               float
    CDi:              float       # induced drag
    CD_wing:          float       # wing parasite
    CD_fuselage:      float       # fuselage parasite
    CD_interference:  float       # wing-fuselage junction
    CD_misc:          float       # antennas, sensors, etc.
    CD0:              float       # total parasite = sum of above
    CD_total:         float       # CD0 + CDi
    LD_ratio:         float       # L/D = CL/CD_total
    Re_wing:          float
    Re_fuselage:      float
    Cf_wing:          float
    Cf_fuse:          float
    FF_wing_val:      float
    FF_fuse_val:      float

    def summary(self) -> str:
        lines = [
            f"\n{'─'*50}",
            f"  Drag Breakdown  α = {self.alpha_deg:.1f}°",
            f"{'─'*50}",
            f"  CL              : {self.CL:.4f}",
            f"  CDi (induced)   : {self.CDi:.5f}  ({100*self.CDi/self.CD_total:.1f}%)",
            f"  CD_wing         : {self.CD_wing:.5f}  ({100*self.CD_wing/self.CD_total:.1f}%)",
            f"  CD_fuselage     : {self.CD_fuselage:.5f}  ({100*self.CD_fuselage/self.CD_total:.1f}%)",
            f"  CD_interference : {self.CD_interference:.5f}  ({100*self.CD_interference/self.CD_total:.1f}%)",
            f"  CD_misc         : {self.CD_misc:.5f}  ({100*self.CD_misc/self.CD_total:.1f}%)",
            f"  CD0 (parasite)  : {self.CD0:.5f}  ({100*self.CD0/self.CD_total:.1f}%)",
            f"{'─'*50}",
            f"  CD_total        : {self.CD_total:.5f}",
            f"  L/D             : {self.LD_ratio:.2f}",
            f"{'─'*50}",
            f"  Re_wing         : {self.Re_wing:.3e}",
            f"  Cf_wing         : {self.Cf_wing:.5f}",
            f"  FF_wing         : {self.FF_wing_val:.4f}",
            f"  Re_fuselage     : {self.Re_fuselage:.3e}",
            f"  Cf_fuselage     : {self.Cf_fuse:.5f}",
            f"  FF_fuselage     : {self.FF_fuse_val:.4f}",
            f"{'─'*50}",
        ]
        return '\n'.join(lines)


def compute_drag(uav, alpha_deg: float, CL: float, CDi: float) -> DragBreakdown:
    """
    Full drag breakdown using the component buildup method.

    Parameters
    
    uav       : UAV object
    alpha_deg : angle of attack [deg]
    CL        : lift coefficient from LLT
    CDi       : induced drag coefficient from LLT
    """
    wing  = uav.wing
    fuse  = uav.fuselage
    S_ref = wing.area

    Re_w = uav.reynolds_number_wing
    Re_f = uav.reynolds_number_fuselage

    #  Wing 
    Cf_w   = Cf_mixed(Re_w, x_tr=0.15)
    FF_w   = FF_wing(wing.thickness_ratio, wing.sweep_angle, x_c_max=0.30)
    S_wet_w = 2 * 1.02 * S_ref          # both surfaces, ×1.02 for thickness
    Q_w    = 1.0                         # isolated surface
    CD_wing = Cf_w * FF_w * Q_w * S_wet_w / S_ref

    #  Fuselage 
    Cf_f   = Cf_turbulent(Re_f)          # fully turbulent (blunter body)
    FF_f   = FF_fuselage(fuse.fineness_ratio)
    Q_f    = 1.0
    CD_fuse = Cf_f * FF_f * Q_f * fuse.wetted_area / S_ref

    #  Interference (wing–fuselage junction) 
    # Typically 5–10% of (wing + body) parasite for a clean low-wing configuration
    CD_int = 0.06 * (CD_wing + CD_fuse)

    #  Miscellaneous (antenna, payload sensor pod, pusher prop hub) 
    # Estimated as equivalent flat-plate area / S_ref
    CD_misc = 0.0025

    #  Totals 
    CD0      = CD_wing + CD_fuse + CD_int + CD_misc
    CD_total = CD0 + CDi
    LD       = CL / CD_total if CD_total > 0 else 0.0

    return DragBreakdown(
        alpha_deg       = alpha_deg,
        CL              = CL,
        CDi             = CDi,
        CD_wing         = CD_wing,
        CD_fuselage     = CD_fuse,
        CD_interference = CD_int,
        CD_misc         = CD_misc,
        CD0             = CD0,
        CD_total        = CD_total,
        LD_ratio        = LD,
        Re_wing         = Re_w,
        Re_fuselage     = Re_f,
        Cf_wing         = Cf_w,
        Cf_fuse         = Cf_f,
        FF_wing_val     = FF_w,
        FF_fuse_val     = FF_f,
    )


def sweep_drag(uav, alpha_range: np.ndarray, llt_results: dict) -> dict:
    """
    Compute drag breakdown at every angle of attack in alpha_range.
    llt_results: output of lifting_line.sweep_alpha()
    """
    records = []
    for i, alpha in enumerate(alpha_range):
        db = compute_drag(uav, alpha, llt_results['CL'][i], llt_results['CDi'][i])
        records.append(db)

    return {
        'alpha':      alpha_range,
        'CL':         np.array([r.CL       for r in records]),
        'CDi':        np.array([r.CDi      for r in records]),
        'CD_wing':    np.array([r.CD_wing  for r in records]),
        'CD_fuselage':np.array([r.CD_fuselage for r in records]),
        'CD_int':     np.array([r.CD_interference for r in records]),
        'CD_misc':    np.array([r.CD_misc  for r in records]),
        'CD0':        np.array([r.CD0      for r in records]),
        'CD_total':   np.array([r.CD_total for r in records]),
        'LD':         np.array([r.LD_ratio for r in records]),
        'records':    records,
    }
