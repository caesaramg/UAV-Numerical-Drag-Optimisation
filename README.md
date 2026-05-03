# UAV Aerodynamic Analysis — Drag Optimisation for Low-Altitude Surveillance

A Python implementation of classical aerodynamic theory applied to a fixed-wing surveillance UAV. The analysis combines **Prandtl's Lifting Line Theory** for induced drag with a **component drag buildup** method (Raymer) for parasite drag, producing a full drag polar, L/D characterisation, and geometry optimisation.

Motivated by my role as Structural Lead of the UoM UAV Society, where real design decisions about wing geometry required a deeper understanding of the aerodynamic trade-offs than commercial simulation tools provide.

---

## Background & Motivation

Fixed-wing UAVs for surveillance missions must balance two competing demands: **endurance** (requiring high L/D, favouring high aspect ratio wings) and **portability / structural feasibility** (constraining wingspan). This analysis quantifies that trade-off from first principles.

Rather than treating CFD software as a black box, this project implements the underlying mathematics in Python — the same philosophy applied in the companion [Numerical Methods / FEM repository](https://github.com/caesaramg/Numerical-Methods-Boundary-Problem).

---

## Aircraft Configuration

| Parameter              | Value           | Notes                                      |
|------------------------|-----------------|--------------------------------------------|
| Airfoil                | NACA 2412       | 2% camber, 12% t/c — good low-Re behaviour |
| Wingspan               | 1.80 m          | Compact for field portability              |
| Root chord             | 0.28 m          |                                            |
| Taper ratio λ          | 0.60            | Reduces induced drag vs rectangular wing   |
| Aspect ratio AR        | 8.04            |                                            |
| Wing area S_ref        | 0.403 m²        |                                            |
| Fuselage length        | 1.20 m          | Pod-and-boom configuration                 |
| Fuselage diameter      | 0.12 m          | Fineness ratio = 10                        |
| Cruise speed           | 22 m/s (79 km/h)|                                            |
| Operating altitude     | 150 m AGL       | ISA atmosphere model                       |
| All-up mass            | 2.50 kg         |                                            |
| Wing Reynolds number   | 3.4 × 10⁵       | Mixed laminar/turbulent transition         |

---

## Theory

### 1. Prandtl's Lifting Line Theory (LLT)

The spanwise circulation distribution Γ(y) is expanded as a Fourier series using the cosine substitution y = −(b/2)cos(θ):

$$\Gamma(\theta) = 2bV_\infty \sum_{n=1}^{N} A_n \sin(n\theta)$$

The fundamental LLT equation at N control points θᵢ gives a linear system:

$$\sum_n A_n \sin(n\theta_i) \left[\frac{4b}{c_i a_0} + \frac{n}{\sin\theta_i}\right] = \alpha - \alpha_{L0}$$

Solving for coefficients Aₙ yields:

- **Lift coefficient:** $C_L = \pi \cdot AR \cdot A_1$
- **Induced drag:** $C_{Di} = C_L^2 / (\pi \cdot AR \cdot e)$
- **Span efficiency:** $e = 1 / (1 + \delta)$ where $\delta = \sum_{n \geq 2} n(A_n/A_1)^2$

An elliptic load distribution gives e = 1 (the theoretical ideal). The tapered wing achieves e ≈ 0.975.

### 2. Component Drag Buildup (Raymer Method)

Parasite drag is summed across all components:

$$C_{D0} = \sum_k \frac{C_{f,k} \cdot FF_k \cdot Q_k \cdot S_{wet,k}}{S_{ref}}$$

Where:
- **Cf** — skin friction coefficient (mixed laminar/turbulent via Prandtl–Schlichting, transition at x/c ≈ 15%)
- **FF** — form factor accounting for thickness and sweep (Raymer §12.4)
- **Q** — interference factor (1.0 for isolated surfaces, adjusted at junctions)
- **S_wet** — wetted area of each component

Components: wing (both surfaces), fuselage (prolate spheroid approximation), wing-fuselage interference, miscellaneous (sensor pod, antenna).

### 3. Optimisation

- **Aspect ratio sweep:** Wing area fixed, AR varied from 4–18. Minimum total drag occurs at the crossover between decreasing CDi and increasing CD0 (more wetted area). Optimal AR ≈ 9.5.
- **Taper ratio sweep:** AR fixed, λ varied from 0.2–1.0. Optimal λ ≈ 0.36 maximises span efficiency. Note: very low taper reduces structural efficiency, so the practical design point is λ ≈ 0.5–0.6.

---

## Results

### Cruise Drag Breakdown (α = 0.31°, CL = 0.208)

| Component          | CD        | % of total |
|--------------------|-----------|------------|
| Wing (skin friction)| 0.01602  | 60.7%      |
| Fuselage           | 0.00484   | 18.4%      |
| Interference       | 0.00125   | 4.7%       |
| Miscellaneous      | 0.00250   | 9.5%       |
| **CD0 (parasite)** | **0.02461** | **93.3%** |
| Induced (CDi)      | 0.00177   | 6.7%       |
| **CD total**       | **0.02638** | **100%**  |
| **L/D at cruise**  | **7.89**  |            |

**Key insight:** At cruise, parasite drag dominates (93%) because the UAV operates at low CL (0.208). Peak L/D of **15.8** is achieved at α ≈ 6.9° where induced and parasite drag are better balanced. This suggests the UAV could cruise at a slightly higher angle of attack if endurance is the priority.

### Optimisation Findings

| Parameter       | Baseline | Optimal | Improvement |
|-----------------|----------|---------|-------------|
| Aspect ratio AR | 8.04     | 9.46    | −0.1% CD    |
| Taper ratio λ   | 0.60     | 0.36    | e: 0.975 → 0.985 |

The AR optimum is shallow — CDi reductions with higher AR are nearly offset by increased skin friction, confirming the wing is already near the aerodynamically efficient region.

---

## Project Structure

```
uav_aero/
├── geometry.py        # UAV dataclasses: Wing, Fuselage, Atmosphere, UAV
├── lifting_line.py    # Prandtl LLT solver (Fourier decomposition)
├── drag_analysis.py   # Component drag buildup (Raymer method)
├── optimisation.py    # AR and taper ratio sweeps with bisection AoA solver
├── visualisation.py   # Matplotlib plot functions (6 figures)
├── main.py            # Orchestrates full analysis, saves all plots
├── requirements.txt
└── results/
    ├── lift_distribution.png
    ├── drag_polar.png
    ├── drag_breakdown.png
    ├── LD_ratio.png
    ├── AR_optimisation.png
    └── taper_optimisation.png
```

---

## Running the Analysis

```bash
pip install -r requirements.txt
python main.py
```

All six plots are saved to `./results/`. Console output prints the full drag breakdown table and optimisation results.

---

## Dependencies

```
numpy>=1.24
matplotlib>=3.7
```

---

## References

1. Anderson, J.D. (2017) *Fundamentals of Aerodynamics*, 6th ed. — Ch. 5 (Lifting Line Theory)
2. Raymer, D.P. (2018) *Aircraft Design: A Conceptual Approach*, 6th ed. — Ch. 12 (Aerodynamics)
3. Schlichting, H. & Gersten, K. (2016) *Boundary Layer Theory* — skin friction models
4. Hoerner, S.F. (1965) *Fluid Dynamic Drag* — component interference factors
5. NACA TN 824 — Characteristics of NACA 2412 airfoil section

---

## Author

**Abdub Guracha** — BEng Aerospace Engineering, University of Manchester  
Structural Lead, UoM UAV Society  
[GitHub](https://github.com/caesaramg) · [LinkedIn](https://www.linkedin.com/in/abdub-guracha/)
