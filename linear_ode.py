"""
Linear ODE Solver — y′ = ay + q(x)
====================================
Analytical solutions and vector-field plots for four special source types:

    1. constant   q(x) = q0
    2. step        q(x) = q0 · H(x − x0)
    3. delta       q(x) = q0 · δ(x − x0)
    4. exponential q(x) = q0 · e^(bx)

Usage (run directly):
    python linear_ode.py                  # all four cases, side-by-side
    python linear_ode.py --case constant  # single case
    python linear_ode.py --case step --a -1 --q0 3 --x0 2
    python linear_ode.py --case exp  --a -1 --b -1   # resonance demo

Requires:  numpy  matplotlib  scipy
Optional:  vector_field.py in the same folder (adds slope-field panel)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import argparse, os, sys


# ── analytical solutions ──────────────────────────────────────────────────────

def solve_constant(x, a, y0, q0):
    """y′ = ay + q0   →   y = (y0 + q0/a)eᵃˣ − q0/a   (a≠0)"""
    if abs(a) < 1e-10:
        return y0 + q0 * x                          # a=0: pure integration
    return (y0 + q0 / a) * np.exp(a * x) - q0 / a


def solve_step(x, a, y0, q0, x0):
    """y′ = ay + q0·H(x−x0)"""
    x = np.asarray(x, float)
    y = np.where(x < x0, y0 * np.exp(a * x), np.nan)
    mask = x >= x0
    if abs(a) < 1e-10:
        y[mask] = y0 * np.exp(a * x[mask]) + q0 * (x[mask] - x0)
    else:
        y[mask] = (y0 * np.exp(a * x[mask])
                   + (q0 / a) * (np.exp(a * (x[mask] - x0)) - 1))
    return y


def solve_delta(x, a, y0, q0, x0):
    """y′ = ay + q0·δ(x−x0)  → jump of q0 at x0"""
    x = np.asarray(x, float)
    before = y0 * np.exp(a * x)
    y0_after = y0 * np.exp(a * x0) + q0        # new IC after kick
    after  = y0_after * np.exp(a * (x - x0))
    return np.where(x < x0, before, after)


def solve_exponential(x, a, y0, q0, b):
    """y′ = ay + q0·eᵇˣ
    b≠a: y = y0·eᵃˣ + q0/(b−a)·(eᵇˣ − eᵃˣ)
    b=a: y = (y0 + q0·x)·eᵃˣ   [resonance]
    """
    x = np.asarray(x, float)
    ea = np.exp(a * x)
    if abs(b - a) < 1e-8:                        # resonance
        return (y0 + q0 * x) * ea
    return y0 * ea + q0 / (b - a) * (np.exp(b * x) - ea)


# ── source functions (for vector field) ──────────────────────────────────────

def q_constant(x, q0, **_):         return np.full_like(np.asarray(x, float), q0)
def q_step(x, q0, x0, **_):        return np.where(np.asarray(x) >= x0, q0, 0.0)
def q_delta(x, q0, x0, **_):       return np.zeros_like(np.asarray(x, float))  # smooth approx
def q_exponential(x, q0, b, **_):  return q0 * np.exp(b * np.asarray(x, float))


# ── single-case figure ────────────────────────────────────────────────────────

def plot_case(
    case,                       # 'constant' | 'step' | 'delta' | 'exp'
    a      = -1.0,
    y0     = 0.0,
    q0     = 2.0,
    x0     = 2.0,               # step / delta location
    b      = 0.5,               # exponential rate
    x_range= (0.0, 8.0),
    y_range= (-4.0, 6.0),
    n_ic   = 6,                 # extra initial conditions for vector field panel
    with_vf= True,              # show vector-field panel if vector_field.py present
    save_path = None,
):
    xs = np.linspace(*x_range, 800)

    # ── pick solver & source ──────────────────────────────────────────────────
    case = case.lower()
    if case == 'constant':
        y_sol  = solve_constant(xs, a, y0, q0)
        q_vals = q_constant(xs, q0=q0)
        F      = lambda x, y: a * y + q0
        title  = f"y′ = {a}y + {q0}  [constant source]"
        formula= (f"y = ({y0:.1f} + {q0}/{a:.1f})·e^({a}x) − {q0}/{a:.1f}"
                  if abs(a) > 1e-9 else f"y = {y0} + {q0}·x")

    elif case == 'step':
        y_sol  = solve_step(xs, a, y0, q0, x0)
        q_vals = q_step(xs, q0=q0, x0=x0)
        F      = lambda x, y: a * y + q_step(x, q0=q0, x0=x0)
        title  = f"y′ = {a}y + {q0}·H(x−{x0})  [step source]"
        formula= f"y = y0·e^(ax)  [x<{x0}] ;  y0·e^(ax) + (q0/a)(e^(a(x−x0))−1)  [x≥{x0}]"

    elif case == 'delta':
        y_sol  = solve_delta(xs, a, y0, q0, x0)
        q_vals = np.zeros_like(xs)
        q_vals[np.argmin(np.abs(xs - x0))] = q0 * (x_range[1]-x_range[0]) / 15
        F      = lambda x, y: a * y          # delta treated as jump IC
        title  = f"y′ = {a}y + {q0}·δ(x−{x0})  [delta source]"
        formula= f"jump of {q0} at x={x0} ;  y(x0⁺) = y0·e^(a·x0) + {q0}"

    elif case in ('exp', 'exponential'):
        case   = 'exp'
        y_sol  = solve_exponential(xs, a, y0, q0, b)
        q_vals = q_exponential(xs, q0=q0, b=b)
        F      = lambda x, y: a * y + q0 * np.exp(b * x)
        res    = "  ← RESONANCE" if abs(b - a) < 0.05 else ""
        title  = f"y′ = {a}y + {q0}·e^({b}x){res}  [exponential source]"
        formula= (f"y = (y0 + q0·x)·e^(ax)  [resonance: b=a={a}]"
                  if abs(b-a) < 0.05
                  else f"y = y0·e^({a}x) + {q0}/({b}−{a})·(e^({b}x) − e^({a}x))")
    else:
        raise ValueError(f"Unknown case '{case}'. Choose: constant, step, delta, exp")

    # ── check if vector_field.py is available ─────────────────────────────────
    vf_available = False
    if with_vf:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(script_dir, 'vector_field.py')):
            sys.path.insert(0, script_dir)
            try:
                from vector_field import plot_vector_field
                vf_available = True
            except ImportError:
                pass

    # ── layout ────────────────────────────────────────────────────────────────
    ncols  = 2 if vf_available else 1
    fig    = plt.figure(figsize=(13 if vf_available else 7, 5))
    gs     = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.35)
    ax_sol = fig.add_subplot(gs[0, 0])

    # ── solution panel ────────────────────────────────────────────────────────
    ax_sol.set_facecolor('#f9f8f6')
    ax_sol.grid(True, color='#cccccc', lw=0.5, ls='--', alpha=0.6)
    ax_sol.set_axisbelow(True)

    # homogeneous solution
    y_hom = y0 * np.exp(a * xs)
    ax_sol.plot(xs, np.clip(y_hom, *y_range), color='#1D9E75',
                lw=1.2, ls='--', alpha=0.7, label='homogeneous y₀eᵃˣ')

    # source (scaled to fit)
    q_scale = (y_range[1]-y_range[0]) / (np.ptp(q_vals) + 1e-9) * 0.25
    ax_sol.plot(xs, np.clip(q_vals * q_scale, *y_range), color='#D85A30',
                lw=1.2, ls=':', alpha=0.8, label='source q(x) [scaled]')

    # delta spike marker
    if case == 'delta':
        ax_sol.axvline(x0, color='#D85A30', lw=1.5, ls=':', alpha=0.7)
        ax_sol.annotate(f'δ(x−{x0})', xy=(x0, y_range[1]*0.85),
                        fontsize=9, color='#D85A30', ha='left')

    # main solution
    ax_sol.plot(xs, np.clip(y_sol, *y_range), color='#7F77DD',
                lw=2.2, label='solution y(x)')

    # step boundary marker
    if case in ('step', 'delta'):
        ax_sol.axvline(x0, color='#888780', lw=0.8, ls='--', alpha=0.5)

    ax_sol.set_xlim(*x_range)
    ax_sol.set_ylim(*y_range)
    ax_sol.set_xlabel('x', fontsize=12)
    ax_sol.set_ylabel('y', fontsize=12, rotation=0, labelpad=12)
    ax_sol.set_title('Solution curve', fontsize=11, pad=8)
    ax_sol.legend(fontsize=9, loc='upper right', framealpha=0.8)
    ax_sol.text(0.01, 0.01, formula, transform=ax_sol.transAxes,
                fontsize=8, color='#5F5E5A', family='monospace',
                verticalalignment='bottom', wrap=True)

    # ── vector field panel ────────────────────────────────────────────────────
    if vf_available:
        ax_vf = fig.add_subplot(gs[0, 1])

        # overlay multiple solution curves
        y_inits = np.linspace(y_range[0]*0.8, y_range[1]*0.8, n_ic)
        for yi in y_inits:
            if case == 'constant':
                yc = solve_constant(xs, a, yi, q0)
            elif case == 'step':
                yc = solve_step(xs, a, yi, q0, x0)
            elif case == 'delta':
                yc = solve_delta(xs, a, yi, q0, x0)
            else:
                yc = solve_exponential(xs, a, yi, q0, b)
            yc = np.clip(yc, y_range[0], y_range[1])
            ax_vf.plot(xs, yc, color='#7F77DD', lw=1.3, alpha=0.65)

        # call plot_vector_field into this axes using a trick:
        # we'll draw the quiver manually here to honour the existing ax
        _draw_quiver(ax_vf, F, x_range, y_range, density=20)

        if case in ('step', 'delta'):
            ax_vf.axvline(x0, color='#888780', lw=0.8, ls='--', alpha=0.6)

        ax_vf.set_xlim(*x_range)
        ax_vf.set_ylim(*y_range)
        ax_vf.set_xlabel('x', fontsize=12)
        ax_vf.set_ylabel('y', fontsize=12, rotation=0, labelpad=12)
        ax_vf.set_title('Vector field + solution curves', fontsize=11, pad=8)
        ax_vf.set_facecolor('#f9f8f6')
        ax_vf.grid(True, color='#cccccc', lw=0.5, ls='--', alpha=0.6)
        ax_vf.set_axisbelow(True)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


def _draw_quiver(ax, F, x_range, y_range, density=20):
    """Draw a normalised quiver field for dy/dx=F(x,y) on ax."""
    import matplotlib.colors as mcolors
    xs = np.linspace(*x_range, density)
    ys = np.linspace(*y_range, density)
    X, Y = np.meshgrid(xs, ys)
    with np.errstate(all='ignore'):
        dY = np.asarray(F(X, Y), float)
    dY = np.where(np.isfinite(dY), dY, 0.0)
    dX = np.ones_like(dY)
    length = np.sqrt(dX**2 + dY**2)
    length[length == 0] = 1
    dXn, dYn = dX / length, dY / length

    norm  = mcolors.Normalize(vmin=np.percentile(dY, 5),
                               vmax=np.percentile(dY, 95))
    cmap  = plt.get_cmap('RdYlBu_r')
    colors = cmap(norm(dY))

    ax.quiver(X, Y, dXn, dYn,
              color    = colors.reshape(-1, 4),
              angles   = 'xy',
              scale    = density * 1.7,
              width    = 0.002,
              headwidth= 4, headlength=4, headaxislength=3.5,
              alpha    = 0.75)


# ── four-panel overview ───────────────────────────────────────────────────────

def plot_all_cases(a=-1.0, y0=0.0, q0=2.0, x0=2.5, b=0.5,
                   x_range=(0, 8), y_range=(-4, 7), save_path=None):
    """Show all four source types in a 2×2 grid."""
    cases = [
        ('constant',  'Constant  q = q₀',
         lambda x,y: a*y + q0),
        ('step',      'Step  q = q₀·H(x−x₀)',
         lambda x,y: a*y + q_step(x, q0=q0, x0=x0)),
        ('delta',     'Delta  q = q₀·δ(x−x₀)',
         lambda x,y: a*y + 0),
        ('exp',       'Exponential  q = q₀·eᵇˣ',
         lambda x,y: a*y + q0*np.exp(b*np.asarray(x, float))),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#f9f8f6')
    xs = np.linspace(*x_range, 800)

    solvers = {
        'constant': lambda yi: solve_constant(xs, a, yi, q0),
        'step':     lambda yi: solve_step(xs, a, yi, q0, x0),
        'delta':    lambda yi: solve_delta(xs, a, yi, q0, x0),
        'exp':      lambda yi: solve_exponential(xs, a, yi, q0, b),
    }

    y_inits = np.linspace(y_range[0]*0.75, y_range[1]*0.75, 8)

    for ax, (case, subtitle, F) in zip(axes.flat, cases):
        ax.set_facecolor('#f9f8f6')
        ax.grid(True, color='#cccccc', lw=0.5, ls='--', alpha=0.5)
        ax.set_axisbelow(True)

        _draw_quiver(ax, F, x_range, y_range, density=18)

        for yi in y_inits:
            yc = np.clip(solvers[case](yi), *y_range)
            ax.plot(xs, yc, color='#7F77DD', lw=1.3, alpha=0.6)

        # highlighted curve through y0
        yh = np.clip(solvers[case](y0), *y_range)
        ax.plot(xs, yh, color='#7F77DD', lw=2.4, alpha=1.0)

        if case in ('step', 'delta'):
            ax.axvline(x0, color='#888780', lw=0.9, ls='--', alpha=0.7)
            ax.annotate(f'x₀={x0}', xy=(x0+0.1, y_range[1]*0.88),
                        fontsize=8, color='#5F5E5A')

        ax.set_xlim(*x_range); ax.set_ylim(*y_range)
        ax.set_title(subtitle, fontsize=11, pad=6)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10, rotation=0, labelpad=10)

    # shared legend
    handles = [
        Line2D([0],[0], color='#7F77DD', lw=2,   label='solution curves'),
        Line2D([0],[0], color='#7F77DD', lw=1.2, alpha=0.5, label='other ICs'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=10, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"y′ = ay + q(x),   a={a},  y₀={y0},  q₀={q0},  x₀={x0},  b={b}",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ── vector_field.py integration examples ─────────────────────────────────────

def demo_with_vector_field():
    """
    Show how to use vector_field.py directly for each special case.
    Requires vector_field.py in the same directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    try:
        from vector_field import plot_vector_field
    except ImportError:
        print("vector_field.py not found — skipping demo_with_vector_field()")
        return

    configs = [
        dict(
            F       = lambda x, y: -1.0 * y + 2.0,
            x_range = (0, 8), y_range = (-3, 7),
            density = 22, title = "y′ = −y + 2  [constant source, a=−1, q₀=2]",
        ),
        dict(
            F       = lambda x, y: -1.0 * y + 2.0 * np.where(np.asarray(x) >= 2.5, 1.0, 0.0),
            x_range = (0, 8), y_range = (-3, 7),
            density = 22, title = "y′ = −y + 2·H(x−2.5)  [step source]",
        ),
        dict(
            F       = lambda x, y: -1.0 * y,            # delta = homogeneous between kicks
            x_range = (0, 8), y_range = (-3, 7),
            density = 22, title = "y′ = −y  [between delta kicks at x₀=2.5]",
        ),
        dict(
            F       = lambda x, y: -1.0 * y + 2.0 * np.exp(0.5 * np.asarray(x, float)),
            x_range = (0, 5), y_range = (-3, 20),
            density = 22, title = "y′ = −y + 2·e^(0.5x)  [exponential source]",
        ),
    ]

    for cfg in configs:
        print(f"Plotting: {cfg['title']}")
        plot_vector_field(**cfg)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Solve and plot y′ = ay + q(x) for special source types.")
    parser.add_argument('--case', default='all',
        choices=['all','constant','step','delta','exp','vf_demo'],
        help="Which case to plot (default: all)")
    parser.add_argument('--a',   type=float, default=-1.0, help="Growth rate a")
    parser.add_argument('--y0',  type=float, default=0.0,  help="Initial condition y(0)")
    parser.add_argument('--q0',  type=float, default=2.0,  help="Source amplitude q0")
    parser.add_argument('--x0',  type=float, default=2.5,  help="Step/delta location x0")
    parser.add_argument('--b',   type=float, default=0.5,  help="Exponential rate b")
    parser.add_argument('--xmin',type=float, default=0.0)
    parser.add_argument('--xmax',type=float, default=8.0)
    parser.add_argument('--ymin',type=float, default=-4.0)
    parser.add_argument('--ymax',type=float, default=7.0)
    parser.add_argument('--save', default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    kw = dict(a=args.a, y0=args.y0, q0=args.q0, x0=args.x0, b=args.b,
              x_range=(args.xmin, args.xmax), y_range=(args.ymin, args.ymax),
              save_path=args.save)

    if args.case == 'all':
        plot_all_cases(**kw)
    elif args.case == 'vf_demo':
        demo_with_vector_field()
    else:
        plot_case(args.case, **kw)
