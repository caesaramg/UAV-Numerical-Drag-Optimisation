"""
geometry.py
-----------
Parametric UAV geometry definitions and ISA atmosphere model.

UAV: Fixed-wing surveillance platform configured for low-altitude (~150m) operations.
Airfoil: NACA 2412 (2% camber, max at 40% chord, 12% thickness) — a common choice
for UAVs due to its favourable low-Reynolds-number characteristics and gentle stall.
"""

import numpy as np
from dataclasses import dataclass, field

# Atmosphere

@dataclass
class Atmosphere:
    """
    International Standard Atmosphere (ISA) model up to 11 km (troposphere).
    All properties computed analytically from altitude.
    """
    altitude: float = 150.0   # metres above sea level

    @property
    def temperature(self) -> float:
        """Static temperature [K] using ISA lapse rate 6.5 K/km."""
        return 288.15 - 0.0065 * self.altitude

    @property
    def pressure(self) -> float:
        """Static pressure [Pa]."""
        return 101325.0 * (self.temperature / 288.15) ** 5.2561

    @property
    def density(self) -> float:
        """Air density [kg/m³] from ideal gas law."""
        return self.pressure / (287.05 * self.temperature)

    @property
    def viscosity(self) -> float:
        """Dynamic viscosity [Pa·s] via Sutherland's law."""
        T = self.temperature
        return 1.458e-6 * T ** 1.5 / (T + 110.4)

    @property
    def speed_of_sound(self) -> float:
        """Speed of sound [m/s]."""
        return np.sqrt(1.4 * 287.05 * self.temperature)

    def __str__(self):
        return (f"Atmosphere @ {self.altitude:.0f} m | "
                f"T={self.temperature:.1f} K  ρ={self.density:.4f} kg/m³  "
                f"μ={self.viscosity:.2e} Pa·s")

# Wing


@dataclass
class Wing:
    """
    Trapezoidal wing geometry.

    Defaults represent a compact fixed-wing surveillance UAV (Skywalker X8 class):
      span = 1.8 m,  root chord = 0.28 m,  taper ratio = 0.60
    giving AR ≈ 8.0 and S_ref ≈ 0.403 m².
    """
    span:          float = 1.80   # b  [m]
    root_chord:    float = 0.28   # cr [m]
    taper_ratio:   float = 0.60   # λ = ct/cr  [-]
    sweep_angle:   float = 0.0    # Λ quarter-chord [deg]
    dihedral:      float = 3.0    # Γ [deg]

    # NACA 2412 aerodynamic properties
    airfoil:            str   = 'NACA2412'
    thickness_ratio:    float = 0.12   # t/c
    alpha_zero_lift:    float = -2.1   # α_L0 [deg]  (from NACA 2412 data)
    a0:                 float = 2 * np.pi  # 2D lift curve slope [rad⁻¹] (thin airfoil)

    # derived geometry

    @property
    def tip_chord(self) -> float:
        return self.root_chord * self.taper_ratio

    @property
    def mean_aerodynamic_chord(self) -> float:
        """MAC for a trapezoidal wing."""
        cr, ct = self.root_chord, self.tip_chord
        return (2/3) * cr * (1 + self.taper_ratio + self.taper_ratio**2) / (1 + self.taper_ratio)

    @property
    def area(self) -> float:
        """Reference (planform) area [m²]."""
        return 0.5 * self.span * (self.root_chord + self.tip_chord)

    @property
    def aspect_ratio(self) -> float:
        return self.span ** 2 / self.area

    def chord_at(self, y: float) -> float:
        """Chord length [m] at spanwise coordinate y ∈ [-b/2, b/2]."""
        eta = abs(2 * y / self.span)         # normalised half-span position
        return self.root_chord * (1 - (1 - self.taper_ratio) * eta)

    def __str__(self):
        return (f"Wing ({self.airfoil}) | span={self.span:.2f} m  "
                f"AR={self.aspect_ratio:.2f}  S={self.area:.4f} m²  "
                f"λ={self.taper_ratio:.2f}")

# Fuselage

@dataclass
class Fuselage:
    """
    Streamlined fuselage — pod-and-boom configuration typical of surveillance UAVs.
    Wetted area approximated as a prolate spheroid.
    """
    length:   float = 1.20   # L  [m]
    diameter: float = 0.12   # d  [m]

    @property
    def fineness_ratio(self) -> float:
        return self.length / self.diameter

    @property
    def wetted_area(self) -> float:
        """Approximation for a streamlined body [m²]."""
        # Prolate spheroid approximation (Raymer §12.3)
        f = self.fineness_ratio
        return np.pi * self.diameter * self.length * (1 - 2/f) ** (2/3) * (
            1 + 1 / (f ** 2 - 1) ** 0.5 * np.arcsin(np.sqrt(1 - (1/f**2))) /
            np.sqrt(1 - 1/f**2)
        ) if f > 1 else np.pi * self.diameter ** 2

    def __str__(self):
        return (f"Fuselage | L={self.length:.2f} m  d={self.diameter:.3f} m  "
                f"f/r={self.fineness_ratio:.1f}  S_wet={self.wetted_area:.4f} m²")


# Complete UAV


@dataclass
class UAV:
    """
    Aggregated UAV model combining wing, fuselage, and flight conditions.
    Default configuration: low-altitude fixed-wing surveillance UAV.
    """
    wing:         Wing       = field(default_factory=Wing)
    fuselage:     Fuselage   = field(default_factory=Fuselage)
    atmosphere:   Atmosphere = field(default_factory=lambda: Atmosphere(altitude=150.0))
    cruise_speed: float      = 22.0   # V∞ [m/s]  — typical for this class
    mass:         float      = 2.5    # m  [kg]   — all-up weight

    @property
    def dynamic_pressure(self) -> float:
        """q∞ = ½ρV² [Pa]."""
        return 0.5 * self.atmosphere.density * self.cruise_speed ** 2

    @property
    def reynolds_number_wing(self) -> float:
        """Re based on mean aerodynamic chord."""
        return (self.atmosphere.density * self.cruise_speed *
                self.wing.mean_aerodynamic_chord / self.atmosphere.viscosity)

    @property
    def reynolds_number_fuselage(self) -> float:
        return (self.atmosphere.density * self.cruise_speed *
                self.fuselage.length / self.atmosphere.viscosity)

    @property
    def mach_number(self) -> float:
        return self.cruise_speed / self.atmosphere.speed_of_sound

    @property
    def lift_required(self) -> float:
        """Lift force [N] required for straight-and-level flight (L = W)."""
        return self.mass * 9.81

    @property
    def CL_cruise(self) -> float:
        """C_L required at cruise conditions."""
        return self.lift_required / (self.dynamic_pressure * self.wing.area)

    def __str__(self):
        return (
            f"\n{'='*55}\n"
            f"  UAV Configuration\n"
            f"{'='*55}\n"
            f"  {self.wing}\n"
            f"  {self.fuselage}\n"
            f"  {self.atmosphere}\n"
            f"  Cruise speed : {self.cruise_speed:.1f} m/s\n"
            f"  Mass         : {self.mass:.2f} kg\n"
            f"  q∞           : {self.dynamic_pressure:.2f} Pa\n"
            f"  Re (wing)    : {self.reynolds_number_wing:.2e}\n"
            f"  Mach         : {self.mach_number:.4f}\n"
            f"  CL_cruise    : {self.CL_cruise:.4f}\n"
            f"{'='*55}"
        )
