# ODE Visualisation Toolkit

Two companion scripts for visualising first-order ordinary differential equations.

| File | Purpose |
|---|---|
| `vector_field.py` | Vector / slope field for **any** ODE `dy/dx = F(x, y)` |
| `linear_ode.py` | Analytical solutions + vector fields for `y′ = ay + q(x)` |

Place both files in the same directory. `linear_ode.py` auto-detects `vector_field.py`
and adds a slope-field panel whenever it is present.

---

## Requirements

```bash
pip install numpy matplotlib scipy
```

---

## `vector_field.py` — General slope field plotter

### What it does

Draws a **quiver (arrow) plot** for any ODE of the form `dy/dx = F(x, y)`.

- Every arrow points in the direction a solution curve travels at that point.
- Arrow **length is normalised** — all arrows are the same size; slope magnitude
  is encoded in **colour** (colorbar on the right).
- **Solution curves** are overlaid automatically, integrated with `scipy` RK45
  forward and backward from the horizontal midpoint.
- **Equilibrium lines** (rows where `F ≈ 0` everywhere) are marked as dashed
  orange lines.
- x-axis ticks are auto-formatted as multiples of π when the range spans π-intervals.

---

### CLI usage

```bash
# default example: dy/dx = 2·cos(x)·cos(y)
python vector_field.py

# built-in named examples
python vector_field.py coscos      # dy/dx = 2·cos(x)·cos(y)
python vector_field.py logistic    # dy/dx = y(1 − y)
python vector_field.py saddle      # dy/dx = x / y
python vector_field.py nonlinear   # dy/dx = sin(x) − y
```

---

### Library usage

Import `plot_vector_field` and pass any Python callable as `F`:

```python
from vector_field import plot_vector_field
import numpy as np

plot_vector_field(
    F       = lambda x, y: x - y**2,
    x_range = (-3, 3),
    y_range = (-3, 3),
    density = 25,
    title   = "dy/dx = x − y²",
)
```

`F` receives **NumPy arrays** for both `x` and `y` — use `np.*` functions throughout.

---

### Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `F` | `callable(x, y)` | — | Right-hand side of `dy/dx = F(x, y)`. Must accept NumPy arrays. |
| `x_range` | `(float, float)` | `(−4π, 4π)` | Horizontal axis limits |
| `y_range` | `(float, float)` | `(−2.5, 2.5)` | Vertical axis limits |
| `density` | `int` | `25` | Grid points per axis. Higher → denser arrows, slower render. |
| `n_curves` | `int` | `8` | Number of solution curves to overlay |
| `show_grid` | `bool` | `True` | Background grid lines |
| `cmap` | `str` | `"RdYlBu_r"` | Matplotlib colormap for arrow colours |
| `title` | `str \| None` | auto | Plot title. `None` tries to extract from lambda source. |
| `figsize` | `(float, float)` | `(11, 6)` | Figure size in inches |
| `save_path` | `str \| None` | `None` | Save to file instead of displaying |

Returns `(fig, ax)` — the matplotlib Figure and Axes objects for further customisation.

---

### Built-in examples

```python
from vector_field import plot_vector_field, EXAMPLES

# list available examples
print(list(EXAMPLES.keys()))
# ['coscos', 'logistic', 'saddle', 'nonlinear']

# run one
plot_vector_field(**EXAMPLES['logistic'])
```

---

### Custom equation examples

#### Autonomous (F depends only on y)

```python
# dy/dx = y² − 1  (two equilibria at y = ±1)
plot_vector_field(
    F       = lambda x, y: y**2 - 1,
    x_range = (-4, 4),
    y_range = (-2.5, 2.5),
    title   = "dy/dx = y² − 1",
)
```

#### Separable

```python
# dy/dx = x·y  (solutions: y = C·e^(x²/2))
plot_vector_field(
    F       = lambda x, y: x * y,
    x_range = (-3, 3),
    y_range = (-3, 3),
    title   = "dy/dx = x·y",
)
```

#### Trigonometric / periodic

```python
# dy/dx = sin(x)·cos(y)
plot_vector_field(
    F       = lambda x, y: np.sin(x) * np.cos(y),
    x_range = (-2 * np.pi, 2 * np.pi),
    y_range = (-np.pi, np.pi),
    density = 28,
    title   = "dy/dx = sin(x)·cos(y)",
)
```

#### Stiff / rapidly changing slope

```python
# dy/dx = −10·y + 10  (fast decay to y=1)
plot_vector_field(
    F       = lambda x, y: -10 * y + 10,
    x_range = (0, 2),
    y_range = (-1, 4),
    density = 20,
    n_curves= 6,
    title   = "dy/dx = −10y + 10",
)
```

#### Saving to a file

```python
fig, ax = plot_vector_field(
    F         = lambda x, y: np.sin(y) - x,
    x_range   = (-3, 3),
    y_range   = (-4, 4),
    save_path = "my_field.png",
)
```

#### Post-processing the returned axes

```python
import matplotlib.pyplot as plt

fig, ax = plot_vector_field(
    F       = lambda x, y: -y + np.sin(x),
    x_range = (-6, 6),
    y_range = (-2, 2),
)

# add a custom annotation
ax.annotate("stable limit cycle", xy=(0, 0), fontsize=11, color="red")
plt.show()
```

---

## `linear_ode.py` — Solver for y′ = ay + q(x)

### Mathematical background

The general first-order linear ODE `y′ − ay = q(x)` is solved via an
integrating factor `μ = e^(−ax)`, giving the **Duhamel formula**:

```
y(x) = y₀·eᵃˣ + ∫₀ˣ e^(a(x−t)) q(t) dt
        ↑                ↑
   homogeneous       particular (convolution)
```

The homogeneous part `y₀·eᵃˣ` decays when `a < 0` (stable) or grows when
`a > 0` (unstable). The particular part accumulates the source's effect.

---

### The four special source types

#### 1. Constant — `q(x) = q₀`

```
y(x) = (y₀ + q₀/a)·eᵃˣ − q₀/a        (a ≠ 0)
y(x) = y₀ + q₀·x                       (a = 0)
```

- Steady state `y* = −q₀/a` exists only when `a < 0`.
- All solutions converge to `y*` regardless of initial condition when `a < 0`.

#### 2. Step — `q(x) = q₀·H(x − x₀)`

```
y(x) = y₀·eᵃˣ                                      for x < x₀
y(x) = y₀·eᵃˣ + (q₀/a)·(e^(a(x−x₀)) − 1)         for x ≥ x₀
```

- Solution is **continuous** at `x₀` (no jump in value).
- The **slope** jumps by `q₀` at `x₀` — a kink, not a break.

#### 3. Delta — `q(x) = q₀·δ(x − x₀)`

```
y(x) = y₀·eᵃˣ                                      for x < x₀
y(x) = (y₀·e^(a·x₀) + q₀)·e^(a(x−x₀))            for x ≥ x₀
```

- The solution **jumps by exactly `q₀`** at `x₀`.
- After the kick, free homogeneous dynamics resume from the new value.
- The vector field shows the homogeneous field `y′ = ay`; the jump is
  an instantaneous shift in initial condition.

#### 4. Exponential — `q(x) = q₀·eᵇˣ`

```
b ≠ a:  y(x) = y₀·eᵃˣ + q₀/(b−a)·(eᵇˣ − eᵃˣ)
b = a:  y(x) = (y₀ + q₀·x)·eᵃˣ                   [resonance]
```

- **Resonance** occurs when `b = a`: the source matches the natural
  frequency of the system and the solution grows as `x·eᵃˣ` — a
  polynomial envelope on top of the exponential.

---

### CLI usage

```bash
# 2×2 overview of all four source types (default parameters)
python linear_ode.py

# single case with defaults
python linear_ode.py --case constant
python linear_ode.py --case step
python linear_ode.py --case delta
python linear_ode.py --case exp

# tune every parameter
python linear_ode.py --case constant --a -0.5 --q0 3 --y0 1
python linear_ode.py --case step     --a -1   --q0 2 --x0 3  --xmax 10
python linear_ode.py --case delta    --a -2   --q0 5 --x0 1
python linear_ode.py --case exp      --a -1   --q0 2 --b 0.5
python linear_ode.py --case exp      --a -1   --q0 2 --b -1   # resonance: b = a

# run the four vector_field.py integration demos sequentially
python linear_ode.py --case vf_demo

# save to file instead of showing
python linear_ode.py --case step --save step_field.png
```

#### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--case` | `all` | `all`, `constant`, `step`, `delta`, `exp`, `vf_demo` |
| `--a` | `-1.0` | Growth/decay rate |
| `--y0` | `0.0` | Initial condition y(0) |
| `--q0` | `2.0` | Source amplitude |
| `--x0` | `2.5` | Location of step / delta (ignored for constant and exp) |
| `--b` | `0.5` | Exponential source rate (exp case only) |
| `--xmin` | `0.0` | Left edge of plot |
| `--xmax` | `8.0` | Right edge of plot |
| `--ymin` | `-4.0` | Bottom edge of plot |
| `--ymax` | `7.0` | Top edge of plot |
| `--save` | — | Filename to save figure (omit to display interactively) |

---

### Library usage

#### Analytical solvers

Each solver takes a NumPy array `x` and returns the corresponding `y` values.

```python
import numpy as np
from linear_ode import (
    solve_constant,
    solve_step,
    solve_delta,
    solve_exponential,
)

xs = np.linspace(0, 8, 500)

# constant source
y = solve_constant(xs, a=-1, y0=0, q0=2)

# step source switching on at x=2.5
y = solve_step(xs, a=-1, y0=0, q0=2, x0=2.5)

# delta kick at x=2.5
y = solve_delta(xs, a=-1, y0=0, q0=2, x0=2.5)

# exponential source (normal case)
y = solve_exponential(xs, a=-1, y0=0, q0=2, b=0.5)

# exponential source (resonance: b == a)
y = solve_exponential(xs, a=-1, y0=0, q0=2, b=-1.0)
```

#### Source functions

These return the value of `q(x)` at an array of x values and are also
suitable for passing directly into `plot_vector_field` as part of `F`:

```python
from linear_ode import q_constant, q_step, q_delta, q_exponential

q = q_constant(xs, q0=2)
q = q_step(xs, q0=2, x0=2.5)
q = q_exponential(xs, q0=2, b=0.5)
```

#### Plotting a single case

```python
from linear_ode import plot_case

# solution curve only (vector_field.py not present)
fig = plot_case('constant', a=-1, y0=0, q0=2)

# solution curve + vector field side by side (vector_field.py present)
fig = plot_case('step', a=-0.5, y0=1, q0=3, x0=3, x_range=(0, 10))

# save instead of showing
fig = plot_case('exp', a=-1, q0=2, b=0.5, save_path='exp_case.png')
```

`plot_case` parameter reference:

| Parameter | Default | Description |
|---|---|---|
| `case` | — | `'constant'`, `'step'`, `'delta'`, `'exp'` |
| `a` | `-1.0` | Growth/decay rate |
| `y0` | `0.0` | Initial condition y(0) |
| `q0` | `2.0` | Source amplitude |
| `x0` | `2.0` | Step / delta location |
| `b` | `0.5` | Exponential source rate |
| `x_range` | `(0, 8)` | Horizontal plot limits |
| `y_range` | `(-4, 6)` | Vertical plot limits |
| `n_ic` | `6` | Number of additional initial conditions in the vector field panel |
| `with_vf` | `True` | Include vector field panel if `vector_field.py` is available |
| `save_path` | `None` | Save to file instead of displaying |

#### Plotting all four cases at once

```python
from linear_ode import plot_all_cases

# default parameters
plot_all_cases()

# custom parameters
plot_all_cases(
    a       = -0.5,
    y0      = 1.0,
    q0      = 3.0,
    x0      = 3.0,
    b       = 0.3,
    x_range = (0, 12),
    y_range = (-5, 10),
    save_path = "overview.png",
)
```

---

### Using `vector_field.py` directly for linear ODE cases

`linear_ode.py` calls `vector_field.py` automatically when both are in the
same folder. You can also call `plot_vector_field` directly to get the full
standalone plot with colorbar and equilibrium detection:

```python
import numpy as np
from vector_field import plot_vector_field

# ── constant source ───────────────────────────────────────────────────────────
plot_vector_field(
    F       = lambda x, y: -1.0 * y + 2.0,
    x_range = (0, 8),
    y_range = (-3, 7),
    density = 22,
    title   = "y′ = −y + 2  [constant, a=−1, q₀=2]",
)

# ── step source H(x − 2.5) ───────────────────────────────────────────────────
plot_vector_field(
    F       = lambda x, y: -y + 2.0 * np.where(np.asarray(x) >= 2.5, 1.0, 0.0),
    x_range = (0, 8),
    y_range = (-3, 7),
    density = 22,
    title   = "y′ = −y + 2·H(x−2.5)  [step source]",
)

# ── delta source (homogeneous field between kicks) ────────────────────────────
# The vector field shows y′ = ay; the delta effect is a vertical jump in the
# solution curve at x₀, handled by linear_ode.solve_delta.
plot_vector_field(
    F       = lambda x, y: -1.0 * y,
    x_range = (0, 8),
    y_range = (-3, 7),
    density = 22,
    title   = "y′ = −y  [homogeneous field between delta kicks]",
)

# ── exponential source ────────────────────────────────────────────────────────
plot_vector_field(
    F       = lambda x, y: -y + 2.0 * np.exp(0.5 * np.asarray(x, float)),
    x_range = (0, 5),
    y_range = (-3, 20),
    density = 22,
    title   = "y′ = −y + 2·e^(0.5x)  [exponential source]",
)

# ── resonance: b = a ──────────────────────────────────────────────────────────
# b = a = −1, so q(x) = 2·e^(−x) matches the homogeneous decay rate.
# Solution grows as x·e^(−x) — observe arrows steepening near x=0.
plot_vector_field(
    F       = lambda x, y: -y + 2.0 * np.exp(-np.asarray(x, float)),
    x_range = (0, 8),
    y_range = (-1, 6),
    density = 22,
    title   = "y′ = −y + 2·e^(−x)  [resonance: b = a = −1]",
)
```

---

### Worked examples — custom source functions

You can pass any callable `F(x, y)` for non-standard sources. The requirement
is that `F` accepts NumPy arrays for both arguments.

#### Sinusoidal source

```python
# y′ = −y + sin(x)
plot_vector_field(
    F       = lambda x, y: -y + np.sin(x),
    x_range = (-2 * np.pi, 2 * np.pi),
    y_range = (-2, 2),
    title   = "y′ = −y + sin(x)",
)
```

#### Piecewise source (multiple steps)

```python
# Source switches on at x=2, doubles at x=5, off at x=8
def piecewise_q(x):
    x = np.asarray(x)
    return np.where(x < 2, 0, np.where(x < 5, 1.0, np.where(x < 8, 2.0, 0.0)))

plot_vector_field(
    F       = lambda x, y: -0.5 * y + piecewise_q(x),
    x_range = (0, 12),
    y_range = (-1, 6),
    title   = "y′ = −0.5y + piecewise q(x)",
)
```

#### Unstable system (a > 0) with a source

```python
# a = +0.5: homogeneous part grows; source competes against it
plot_vector_field(
    F       = lambda x, y: 0.5 * y - 3.0,
    x_range = (0, 6),
    y_range = (-2, 10),
    n_curves= 6,
    title   = "y′ = 0.5y − 3  [unstable, a > 0]",
)
```

#### Gaussian pulse source

```python
# Localised burst of forcing near x = 3
def gaussian_q(x, centre=3.0, width=0.5, amplitude=4.0):
    return amplitude * np.exp(-((np.asarray(x) - centre) / width) ** 2)

plot_vector_field(
    F       = lambda x, y: -y + gaussian_q(x),
    x_range = (0, 8),
    y_range = (-1, 5),
    title   = "y′ = −y + 4·exp(−((x−3)/0.5)²)  [Gaussian pulse]",
)
```

#### Combining analytical solution with vector field

```python
import numpy as np
import matplotlib.pyplot as plt
from vector_field import plot_vector_field
from linear_ode import solve_step

xs = np.linspace(0, 8, 800)
a, y0, q0, x0 = -1.0, 0.5, 3.0, 2.5

fig, ax = plot_vector_field(
    F       = lambda x, y: a * y + q0 * np.where(np.asarray(x) >= x0, 1.0, 0.0),
    x_range = (0, 8),
    y_range = (-2, 8),
    n_curves= 0,            # suppress auto-generated curves
    title   = "Step source — analytical solution overlaid",
)

# overlay the exact analytical solution for y(0) = 0.5
y_exact = solve_step(xs, a, y0, q0, x0)
ax.plot(xs, np.clip(y_exact, -2, 8),
        color='red', lw=2.2, ls='--', label=f'exact  y(0)={y0}', zorder=5)
ax.legend(fontsize=10)
plt.show()
```

---

## Quick reference — which function to call

```
I want to …                                         Use
─────────────────────────────────────────────────────────────────────────────
Plot a slope field for any dy/dx = F(x,y)           plot_vector_field(F, ...)
Plot solution + field for y′ = ay + const           plot_case('constant', ...)
Plot solution + field for y′ = ay + step            plot_case('step', ...)
Plot solution + field for y′ = ay + delta           plot_case('delta', ...)
Plot solution + field for y′ = ay + exp             plot_case('exp', ...)
Compare all four sources side by side               plot_all_cases(...)
Get the y-values of a solution as a NumPy array     solve_constant / solve_step
                                                    solve_delta / solve_exponential
Run the vector_field demos for each linear case     python linear_ode.py --case vf_demo
```

---

## File layout

```
your_project/
├── vector_field.py    ← general slope-field plotter
├── linear_ode.py      ← linear ODE solver (auto-detects vector_field.py)
└── README.md
```

Both files are self-contained and importable. `linear_ode.py` degrades gracefully
when `vector_field.py` is absent — it shows the solution panel only.
