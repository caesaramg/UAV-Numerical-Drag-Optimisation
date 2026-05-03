"""
visualisation.py
----------------
Publication-quality plots for the UAV aerodynamic analysis.

Figures produced:
  1. Spanwise lift distribution vs elliptic ideal
  2. Drag polar (CD vs CL) with operating point
  3. Drag breakdown stacked area chart vs angle of attack
  4. L/D ratio vs angle of attack
  5. Aspect ratio optimisation — CD components vs AR
  6. Taper ratio optimisation — span efficiency vs λ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Consistent style
STYLE = {
    'figure.facecolor':   '#0f0f0f',
    'axes.facecolor':     '#1a1a1a',
    'axes.edgecolor':     '#444',
    'axes.labelcolor':    '#e0e0e0',
    'axes.titlecolor':    '#ffffff',
    'axes.grid':          True,
    'grid.color':         '#2a2a2a',
    'grid.linestyle':     '--',
    'xtick.color':        '#aaa',
    'ytick.color':        '#aaa',
    'text.color':         '#e0e0e0',
    'legend.facecolor':   '#222',
    'legend.edgecolor':   '#444',
    'font.family':        'monospace',
    'font.size':          10,
}

BLUE    = '#4fc3f7'
ORANGE  = '#ff8a65'
GREEN   = '#81c784'
YELLOW  = '#fff176'
PURPLE  = '#ce93d8'
RED     = '#ef9a9a'
TEAL    = '#4db6ac'


def _apply_style():
    plt.rcParams.update(STYLE)



# 1. Spanwise lift distribution


def plot_lift_distribution(llt_result: dict, wing, save_path: str = None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    y           = llt_result['y']
    lift_dist   = llt_result['lift_dist']
    lift_ellip  = llt_result['lift_dist_elliptic']
    alpha       = llt_result['alpha_deg']

    # Mirror to full span
    y_full    = np.concatenate([-y[::-1], y])
    ld_full   = np.concatenate([lift_dist[::-1], lift_dist])
    el_full   = np.concatenate([lift_ellip[::-1], lift_ellip])

    ax.fill_between(y_full, ld_full, alpha=0.3, color=BLUE)
    ax.plot(y_full, ld_full,  color=BLUE,   lw=2.0, label='LLT — Tapered wing')
    ax.plot(y_full, el_full,  color=ORANGE, lw=1.8, ls='--', label='Elliptic ideal')

    ax.set_xlabel('Spanwise position  y  [m]')
    ax.set_ylabel('Local lift loading  $c_\\ell \\cdot c(y)$  [m]')
    ax.set_title(f'Spanwise Lift Distribution  |  α = {alpha:.1f}°  |  AR = {wing.aspect_ratio:.2f}')
    ax.legend()
    ax.set_xlim(-wing.span/2 * 1.05, wing.span/2 * 1.05)
    ax.axvline(0, color='#555', lw=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig



# 2. Drag polar


def plot_drag_polar(sweep: dict, cruise_point: dict = None, save_path: str = None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    CD  = sweep['CD_total']
    CL  = sweep['CL']
    CD0 = sweep['CD0'][0]           # parasite drag (approx constant with alpha)

    ax.plot(CD, CL, color=BLUE, lw=2.2, label='Drag polar')

    # Mark best L/D point
    LD      = CL / CD
    idx_LD  = np.argmax(LD)
    ax.scatter(CD[idx_LD], CL[idx_LD], s=120, color=GREEN, zorder=5,
               label=f'Best L/D = {LD[idx_LD]:.1f}  (CL={CL[idx_LD]:.3f})')

    # Mark cruise operating point
    if cruise_point is not None:
        ax.scatter(cruise_point['CD'], cruise_point['CL'], s=120,
                   color=ORANGE, marker='*', zorder=6,
                   label=f"Cruise  (CL={cruise_point['CL']:.3f})")

    # Parabolic fit reference  CD = CD0 + CL²/(π AR e)
    CL_fit = np.linspace(0, max(CL) * 1.1, 200)
    # estimate AR*e from induced drag slope
    LD_arr  = CL / CD
    e_mean  = np.mean(sweep.get('CDi', CL**2) / (np.pi * 8.0 * CL**2 + 1e-12))

    ax.set_xlabel('Drag coefficient  $C_D$')
    ax.set_ylabel('Lift coefficient  $C_L$')
    ax.set_title('Drag Polar  —  UAV NACA 2412 Wing')
    ax.legend(loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig



# 3. Drag breakdown stacked area chart


def plot_drag_breakdown(sweep: dict, save_path: str = None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    alpha = sweep['alpha']
    labels = ['Induced (CDi)', 'Wing skin friction', 'Fuselage', 'Interference', 'Misc']
    data   = [
        sweep['CDi'],
        sweep['CD_wing'],
        sweep['CD_fuselage'],
        sweep['CD_int'],
        np.full_like(alpha, sweep['CD_misc'][0]),
    ]
    colors = [BLUE, ORANGE, GREEN, PURPLE, YELLOW]

    ax.stackplot(alpha, data, labels=labels, colors=colors, alpha=0.85)

    ax.set_xlabel('Angle of attack  α  [deg]')
    ax.set_ylabel('Drag coefficient  $C_D$')
    ax.set_title('Drag Breakdown by Component  —  UAV at Cruise Speed')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(alpha[0], alpha[-1])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig



# 4. L/D ratio vs AoA


def plot_LD_ratio(sweep: dict, save_path: str = None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    alpha = sweep['alpha']
    LD    = sweep['LD']
    idx   = np.argmax(LD)

    ax.plot(alpha, LD, color=GREEN, lw=2.2)
    ax.axvline(alpha[idx], color=ORANGE, ls='--', lw=1.4,
               label=f'Best L/D at α = {alpha[idx]:.1f}°  →  L/D = {LD[idx]:.2f}')
    ax.scatter(alpha[idx], LD[idx], s=120, color=ORANGE, zorder=5)
    ax.axhline(LD[idx], color='#555', ls=':', lw=0.8)

    ax.set_xlabel('Angle of attack  α  [deg]')
    ax.set_ylabel('Lift-to-Drag ratio  $L/D$')
    ax.set_title('Aerodynamic Efficiency  —  L/D vs Angle of Attack')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig



# 5. Aspect ratio optimisation


def plot_AR_optimisation(opt: dict, save_path: str = None):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    AR   = opt['AR_range']
    ax.plot(AR, opt['CD_total'] * 1e4, color=BLUE,   lw=2.2, label='$C_{D,total}$  (×10⁻⁴)')
    ax.plot(AR, opt['CD0']     * 1e4, color=ORANGE, lw=1.8, ls='--', label='$C_{D0}$ parasite (×10⁻⁴)')
    ax.plot(AR, opt['CDi']     * 1e4, color=GREEN,  lw=1.8, ls='--', label='$C_{Di}$ induced (×10⁻⁴)')

    AR_opt = opt['AR_optimal']
    CD_opt = opt['CD_optimal'] * 1e4
    ax.axvline(AR_opt, color=YELLOW, ls=':', lw=1.4,
               label=f'Optimal AR = {AR_opt:.2f}  →  $C_D$ = {CD_opt:.2f}×10⁻⁴')
    ax.scatter([AR_opt], [CD_opt], s=140, color=YELLOW, zorder=6)

    ax.set_xlabel('Aspect Ratio  AR')
    ax.set_ylabel('Drag Coefficient  (×10⁻⁴)')
    ax.set_title('Aspect Ratio Optimisation  —  Constant Wing Area, Cruise CL')
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig



# 6. Taper ratio optimisation


def plot_taper_optimisation(opt: dict, save_path: str = None):
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    lam = opt['lambda_range']
    ax1.plot(lam, opt['e'],        color=BLUE,   lw=2.2, label='Span efficiency  e')
    ax2.plot(lam, opt['CD_total'] * 1e4, color=ORANGE, lw=1.8, ls='--',
             label='$C_{D,total}$  (×10⁻⁴)')

    lam_opt = opt['lambda_optimal']
    ax1.axvline(lam_opt, color=YELLOW, ls=':', lw=1.4,
                label=f'Optimal λ = {lam_opt:.2f}  (e = {opt["e_optimal"]:.4f})')

    ax1.set_xlabel('Taper ratio  λ = $c_t / c_r$')
    ax1.set_ylabel('Span efficiency  e', color=BLUE)
    ax2.set_ylabel('$C_{D,total}$  (×10⁻⁴)', color=ORANGE)
    ax1.set_title('Taper Ratio Optimisation  —  Fixed AR & Wing Area')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax1.tick_params(axis='y', labelcolor=BLUE)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
