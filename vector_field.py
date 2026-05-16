"""
Vector Field Plotter — dy/dx = F(x, y)
=======================================
Displays a quiver plot (vector field / slope field) for any first-order ODE
of the form  dy/dx = F(x, y).

Usage
-----
Run directly to see the built-in example (y' = 2·cos x·cos y):

    python vector_field.py

Or import and call plot_vector_field() with your own function:

    from vector_field import plot_vector_field
    import numpy as np

    plot_vector_field(
        F       = lambda x, y: x - y**2,
        x_range = (-3, 3),
        y_range = (-3, 3),
        density = 25,
        title   = "dy/dx = x − y²",
    )

Configuration
-------------
F         : callable (x, y) -> float | array
              The right-hand side of  dy/dx = F(x, y).
              Both x and y will be NumPy arrays — use np.* functions.

x_range   : (float, float)   horizontal axis limits, default (-4π, 4π)
y_range   : (float, float)   vertical   axis limits, default (-2.5, 2.5)
density   : int              grid points per axis (arrows), default 25
n_curves  : int              number of solution curves overlaid, default 8
show_grid : bool             show background grid lines, default True
cmap      : str              matplotlib colormap for arrow colouring, default "RdYlBu_r"
title     : str              plot title (None → auto-generated)
figsize   : (float, float)   figure size in inches, default (11, 6)
save_path : str | None       if given, saves the figure instead of showing it
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp


# ── helper ────────────────────────────────────────────────────────────────────

def _auto_title(F):
    """Try to recover a readable title from the function's source."""
    import inspect
    try:
        src = inspect.getsource(F).strip()
        # lambda: grab the right-hand side
        if "lambda" in src:
            rhs = src.split(":", 1)[-1].strip().rstrip(",)")
            return f"dy/dx = {rhs}"
    except Exception:
        pass
    return "dy/dx = F(x, y)"


def _safe_eval(F, x, y):
    """Evaluate F(x, y); replace infinities / NaN with 0 so quiver doesn't crash."""
    with np.errstate(all="ignore"):
        z = np.asarray(F(x, y), dtype=float)
    z = np.where(np.isfinite(z), z, 0.0)
    return z


# ── main function ──────────────────────────────────────────────────────────────

def plot_vector_field(
    F,
    x_range   = (-4 * np.pi, 4 * np.pi),
    y_range   = (-2.5, 2.5),
    density   = 25,
    n_curves  = 8,
    show_grid = True,
    cmap      = "RdYlBu_r",
    title     = None,
    figsize   = (11, 6),
    save_path = None,
):
    """
    Plot the vector / slope field for  dy/dx = F(x, y).

    Parameters
    ----------
    See module docstring for full parameter descriptions.
    """
    x0, x1 = x_range
    y0, y1 = y_range

    # ── 1. build the grid ─────────────────────────────────────────────────────
    xs = np.linspace(x0, x1, density)
    ys = np.linspace(y0, y1, density)
    X, Y = np.meshgrid(xs, ys)

    dY = _safe_eval(F, X, Y)
    dX = np.ones_like(dY)

    # normalise arrow lengths so every arrow has unit length in screen space
    length = np.sqrt(dX**2 + dY**2)
    length = np.where(length == 0, 1, length)
    dX_n, dY_n = dX / length, dY / length

    # colour encodes the signed slope magnitude
    slope_mag = dY   # use raw slope for colour (not normalised)

    # ── 2. figure setup ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor("#f9f8f6")
    ax.set_facecolor("#f9f8f6")

    if show_grid:
        ax.grid(True, color="#cccccc", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    # ── 3. quiver (vector field) ───────────────────────────────────────────────
    norm = mcolors.Normalize(
        vmin=np.percentile(slope_mag, 5),
        vmax=np.percentile(slope_mag, 95),
    )
    colormap = plt.get_cmap(cmap)
    arrow_colors = colormap(norm(slope_mag))

    q = ax.quiver(
        X, Y, dX_n, dY_n,
        color    = arrow_colors.reshape(-1, 4),
        angles   = "xy",
        scale    = density * 1.6,
        width    = 0.0018,
        headwidth= 4,
        headlength     = 4,
        headaxislength = 3.5,
        alpha    = 0.85,
    )

    # ── 4. solution curves ────────────────────────────────────────────────────
    if n_curves > 0:
        # spread initial y-values evenly; integrate forward and backward
        y_inits = np.linspace(y0 * 0.85, y1 * 0.85, n_curves)
        x_mid   = (x0 + x1) / 2      # start all curves at the horizontal midpoint

        for y_init in y_inits:
            for direction, x_span in [(1, (x_mid, x1)), (-1, (x_mid, x0))]:
                try:
                    sol = solve_ivp(
                        fun     = lambda x, y: [_safe_eval(F, x, y[0])],
                        t_span  = x_span,
                        y0      = [y_init],
                        method  = "RK45",
                        dense_output = True,
                        max_step     = (x1 - x0) / 300,
                        rtol    = 1e-5,
                        atol    = 1e-7,
                    )
                    x_plot = sol.t
                    y_plot = sol.y[0]
                    # clip to visible range
                    mask = (y_plot >= y0) & (y_plot <= y1)
                    if mask.sum() > 2:
                        ax.plot(
                            x_plot[mask], y_plot[mask],
                            color     = "#4c4ac0",
                            linewidth = 1.4,
                            alpha     = 0.75,
                            zorder    = 3,
                        )
                except Exception:
                    pass   # skip curves that fail to integrate

    # ── 5. equilibrium lines (y where F(x,y)=0 for typical x) ────────────────
    # just draw the zero-slope reference — horizontal line where F=0 everywhere
    # (detected as columns in dY that are uniformly near zero)
    y_eq_candidates = ys[np.all(np.abs(dY) < 1e-2, axis=1)]
    for ye in y_eq_candidates:
        ax.axhline(ye, color="#e05a2b", linewidth=1.2,
                   linestyle="--", alpha=0.8, zorder=2)

    # ── 6. axes, labels, colour bar ───────────────────────────────────────────
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13, rotation=0, labelpad=12)
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.axvline(0, color="#888", linewidth=0.6)
    ax.tick_params(labelsize=10)

    # π-labelled ticks on x if range spans multiples of π
    span = x1 - x0
    if span > 2 and abs(span / np.pi - round(span / np.pi)) < 0.2:
        pi_ticks = [k * np.pi for k in range(-8, 9)
                    if x0 <= k * np.pi <= x1]
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(
            [("0" if k == 0 else (f"{k}π" if k != 1 else "π"))
             for k in range(-8, 9) if x0 <= k * np.pi <= x1],
            fontsize=10,
        )

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("slope  dy/dx", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plot_title = title if title is not None else _auto_title(F)
    ax.set_title(plot_title, fontsize=14, pad=12)

    # ── 7. legend ─────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#4c4ac0", linewidth=1.6, label="solution curves"),
    ]
    if len(y_eq_candidates):
        handles.append(
            Line2D([0], [0], color="#e05a2b", linewidth=1.2,
                   linestyle="--", label="equilibria")
        )
    ax.legend(handles=handles, fontsize=10, loc="upper right",
              framealpha=0.7, edgecolor="#ccc")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig, ax


# ── examples ──────────────────────────────────────────────────────────────────

EXAMPLES = {
    "coscos": {
        "F"       : lambda x, y: 2 * np.cos(x) * np.cos(y),
        "x_range" : (-4 * np.pi, 4 * np.pi),
        "y_range" : (-2.4, 2.4),
        "density" : 26,
        "title"   : "dy/dx = 2·cos(x)·cos(y)",
    },
    "logistic": {
        "F"       : lambda x, y: y * (1 - y),
        "x_range" : (-1, 6),
        "y_range" : (-0.5, 1.8),
        "density" : 24,
        "title"   : "dy/dx = y(1 − y)  [logistic]",
    },
    "saddle": {
        "F"       : lambda x, y: x / (y + 1e-9),
        "x_range" : (-3, 3),
        "y_range" : (-3, 3),
        "density" : 24,
        "title"   : "dy/dx = x / y  [saddle / hyperbolas]",
    },
    "nonlinear": {
        "F"       : lambda x, y: np.sin(x) - y,
        "x_range" : (-6, 6),
        "y_range" : (-2.5, 2.5),
        "density" : 25,
        "title"   : "dy/dx = sin(x) − y",
    },
}


if __name__ == "__main__":
    import sys

    # pick example from command line, e.g.:  python vector_field.py logistic
    key = sys.argv[1] if len(sys.argv) > 1 else "coscos"

    if key not in EXAMPLES:
        print(f"Unknown example '{key}'. Available: {list(EXAMPLES.keys())}")
        sys.exit(1)

    cfg = EXAMPLES[key]
    print(f"Plotting: {cfg['title']}")
    plot_vector_field(**cfg)
