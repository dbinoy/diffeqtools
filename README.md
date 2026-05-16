# ODE Visualisation Toolkit

Two companion scripts for visualising first-order ordinary differential equations.

| File | Purpose |
|---|---|
| `vector_field.py` | Vector / slope field for **any** ODE $\dfrac{dy}{dx} = F(x,\, y)$ |
| `linear_ode.py` | Analytical solutions + vector fields for $y' = ay + q(x)$ |

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

Draws a **quiver (arrow) plot** for any ODE of the form

$$\frac{dy}{dx} = F(x,\, y)$$

- Every arrow points in the direction a solution curve travels at that point.
- Arrow **length is normalised** — all arrows are the same size; slope magnitude
  is encoded in **colour** (colorbar on the right).
- **Solution curves** are overlaid automatically, integrated with `scipy` RK45
  forward and backward from the horizontal midpoint.
- **Equilibrium lines** (rows where $F \approx 0$ everywhere) are marked as
  dashed orange lines.
- $x$-axis ticks are auto-formatted as multiples of $\pi$ when the range spans
  $\pi$-intervals.

---

### CLI usage

```bash
# default example: dy/dx = 2 cos(x) cos(y)
python vector_field.py

# built-in named examples
python vector_field.py coscos      # dy/dx = 2 cos(x) cos(y)
python vector_field.py logistic    # dy/dx = y(1 - y)
python vector_field.py saddle      # dy/dx = x / y
python vector_field.py nonlinear   # dy/dx = sin(x) - y
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
    title   = "dy/dx = x - y^2",
)
```

`F` receives **NumPy arrays** for both `x` and `y` — use `np.*` functions throughout.

---

### Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `F` | `callable(x, y)` | — | Right-hand side of $\frac{dy}{dx} = F(x,y)$. Must accept NumPy arrays. |
| `x_range` | `(float, float)` | $(-4\pi,\ 4\pi)$ | Horizontal axis limits |
| `y_range` | `(float, float)` | $(-2.5,\ 2.5)$ | Vertical axis limits |
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

print(list(EXAMPLES.keys()))
# ['coscos', 'logistic', 'saddle', 'nonlinear']

plot_vector_field(**EXAMPLES['logistic'])
```

| Key | Equation |
|---|---|
| `coscos` | $\dfrac{dy}{dx} = 2\cos(x)\cos(y)$ |
| `logistic` | $\dfrac{dy}{dx} = y(1 - y)$ |
| `saddle` | $\dfrac{dy}{dx} = \dfrac{x}{y}$ |
| `nonlinear` | $\dfrac{dy}{dx} = \sin(x) - y$ |

---

### Custom equation examples

#### Autonomous — $F$ depends only on $y$

Two equilibria at $y = \pm 1$:

$$\frac{dy}{dx} = y^2 - 1$$

```python
plot_vector_field(
    F       = lambda x, y: y**2 - 1,
    x_range = (-4, 4),
    y_range = (-2.5, 2.5),
    title   = "dy/dx = y^2 - 1",
)
```

#### Separable

Solutions are $y = C\, e^{x^2/2}$:

$$\frac{dy}{dx} = x\,y$$

```python
plot_vector_field(
    F       = lambda x, y: x * y,
    x_range = (-3, 3),
    y_range = (-3, 3),
    title   = "dy/dx = x*y",
)
```

#### Trigonometric / periodic

$$\frac{dy}{dx} = \sin(x)\cos(y)$$

```python
plot_vector_field(
    F       = lambda x, y: np.sin(x) * np.cos(y),
    x_range = (-2 * np.pi, 2 * np.pi),
    y_range = (-np.pi, np.pi),
    density = 28,
    title   = "dy/dx = sin(x) cos(y)",
)
```

#### Stiff / rapidly changing slope

Fast decay to the equilibrium $y^* = 1$:

$$\frac{dy}{dx} = -10\,y + 10$$

```python
plot_vector_field(
    F       = lambda x, y: -10 * y + 10,
    x_range = (0, 2),
    y_range = (-1, 4),
    density = 20,
    n_curves= 6,
    title   = "dy/dx = -10y + 10",
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
ax.annotate("stable limit cycle", xy=(0, 0), fontsize=11, color="red")
plt.show()
```

---

## `linear_ode.py` — Solver for $y' = ay + q(x)$

### Mathematical background

The general first-order linear ODE

$$y' - ay = q(x)$$

is solved with the integrating factor $\mu(x) = e^{-ax}$, which converts the
left-hand side into an exact derivative:

$$\frac{d}{dx}\!\left[e^{-ax}\, y\right] = e^{-ax}\, q(x)$$

Integrating both sides and applying the initial condition $y(0) = y_0$ gives
the **Duhamel (variation of parameters) formula**:

$$\boxed{y(x) \;=\; \underbrace{y_0\, e^{ax}}_{\text{homogeneous}} \;+\; \underbrace{\int_0^x e^{a(x-t)}\, q(t)\; dt}_{\text{particular (convolution)}}}$$

The homogeneous part $y_0 e^{ax}$ **decays** when $a < 0$ (stable system) and
**grows** when $a > 0$ (unstable system). The particular integral convolves the
source $q$ with the system's impulse response $e^{ax}$.

---

### The four special source types

#### 1. Constant — $q(x) = q_0$

$$y(x) = \left(y_0 + \frac{q_0}{a}\right)e^{ax} - \frac{q_0}{a} \qquad (a \neq 0)$$

$$y(x) = y_0 + q_0\, x \qquad (a = 0)$$

The **steady state** $y^* = -\dfrac{q_0}{a}$ exists only when $a < 0$.
All solutions converge to $y^*$ regardless of initial condition.

#### 2. Step — $q(x) = q_0\, H(x - x_0)$

$$y(x) = \begin{cases} y_0\, e^{ax} & x < x_0 \\[6pt] y_0\, e^{ax} + \dfrac{q_0}{a}\!\left(e^{a(x - x_0)} - 1\right) & x \geq x_0 \end{cases}$$

- The solution is **continuous** at $x_0$ (no jump in value).
- The **derivative** jumps by $q_0$ at $x_0$ — a kink, not a break.

#### 3. Delta — $q(x) = q_0\, \delta(x - x_0)$

$$y(x) = \begin{cases} y_0\, e^{ax} & x < x_0 \\[6pt] \!\left(y_0\, e^{a x_0} + q_0\right) e^{a(x - x_0)} & x \geq x_0 \end{cases}$$

- The solution **jumps by exactly $q_0$** at $x = x_0$.
- After the instantaneous kick, free homogeneous dynamics resume from
  $y(x_0^+) = y_0 e^{ax_0} + q_0$.

> **Note on the vector field.** Because $\delta(x - x_0)$ is a distribution,
> the slope-field panel shows the homogeneous equation $y' = ay$ everywhere.
> The delta's effect appears in the solution curve as a vertical jump of
> size $q_0$ at $x = x_0$.

#### 4. Exponential — $q(x) = q_0\, e^{bx}$

**Normal case** $b \neq a$:

$$y(x) = y_0\, e^{ax} + \frac{q_0}{b - a}\!\left(e^{bx} - e^{ax}\right)$$

**Resonance** $b = a$ (source frequency matches the system's natural rate):

$$y(x) = \left(y_0 + q_0\, x\right) e^{ax}$$

Resonance produces growth as $x e^{ax}$ — a polynomial envelope on top of
the exponential — rather than pure exponential behaviour.

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

# run all four vector_field.py integration demos sequentially
python linear_ode.py --case vf_demo

# save to file instead of showing
python linear_ode.py --case step --save step_field.png
```

#### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--case` | `all` | `all`, `constant`, `step`, `delta`, `exp`, `vf_demo` |
| `--a` | `-1.0` | Growth/decay rate $a$ |
| `--y0` | `0.0` | Initial condition $y_0 = y(0)$ |
| `--q0` | `2.0` | Source amplitude $q_0$ |
| `--x0` | `2.5` | Location $x_0$ of step / delta (ignored for `constant` and `exp`) |
| `--b` | `0.5` | Exponential source rate $b$ (`exp` case only) |
| `--xmin` | `0.0` | Left edge of plot |
| `--xmax` | `8.0` | Right edge of plot |
| `--ymin` | `-4.0` | Bottom edge of plot |
| `--ymax` | `7.0` | Top edge of plot |
| `--save` | — | Filename to save figure (omit to display interactively) |

---

### Library usage

#### Analytical solvers

Each solver accepts a NumPy array `x` and returns the corresponding $y$ values.

```python
import numpy as np
from linear_ode import (
    solve_constant,
    solve_step,
    solve_delta,
    solve_exponential,
)

xs = np.linspace(0, 8, 500)

# q(x) = q0  =>  y = (y0 + q0/a) e^(ax) - q0/a
y = solve_constant(xs, a=-1, y0=0, q0=2)

# q(x) = q0 H(x - x0)
y = solve_step(xs, a=-1, y0=0, q0=2, x0=2.5)

# q(x) = q0 delta(x - x0)  =>  jump of q0 at x0
y = solve_delta(xs, a=-1, y0=0, q0=2, x0=2.5)

# q(x) = q0 e^(bx),  b != a
y = solve_exponential(xs, a=-1, y0=0, q0=2, b=0.5)

# q(x) = q0 e^(ax),  b == a  =>  resonance: y = (y0 + q0 x) e^(ax)
y = solve_exponential(xs, a=-1, y0=0, q0=2, b=-1.0)
```

#### Source functions

These evaluate $q(x)$ over an array of $x$ values and can be passed directly
into `plot_vector_field` as part of $F$:

```python
from linear_ode import q_constant, q_step, q_exponential

q = q_constant(xs, q0=2)              # q(x) = 2
q = q_step(xs, q0=2, x0=2.5)         # q(x) = 2 H(x - 2.5)
q = q_exponential(xs, q0=2, b=0.5)   # q(x) = 2 e^(0.5x)
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
| `a` | `-1.0` | Growth/decay rate $a$ |
| `y0` | `0.0` | Initial condition $y_0 = y(0)$ |
| `q0` | `2.0` | Source amplitude $q_0$ |
| `x0` | `2.0` | Step / delta location $x_0$ |
| `b` | `0.5` | Exponential source rate $b$ |
| `x_range` | `(0, 8)` | Horizontal plot limits |
| `y_range` | `(-4, 6)` | Vertical plot limits |
| `n_ic` | `6` | Additional initial conditions shown in the vector field panel |
| `with_vf` | `True` | Include vector field panel if `vector_field.py` is available |
| `save_path` | `None` | Save to file instead of displaying |

#### Plotting all four cases at once

```python
from linear_ode import plot_all_cases

plot_all_cases()   # default parameters

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
standalone plot with colorbar and equilibrium detection.
The key is expressing each source as part of the $F(x, y)$ lambda.

#### Constant source

$$y' = -y + 2, \qquad a = -1,\quad q_0 = 2,\quad y^* = 2$$

```python
import numpy as np
from vector_field import plot_vector_field

plot_vector_field(
    F       = lambda x, y: -1.0 * y + 2.0,
    x_range = (0, 8),
    y_range = (-3, 7),
    density = 22,
    title   = "y' = -y + 2  (constant source)",
)
```

#### Step source

$$y' = -y + 2\,H(x - 2.5), \qquad H(x-x_0) = \begin{cases} 0 & x < x_0 \\ 1 & x \geq x_0 \end{cases}$$

```python
plot_vector_field(
    F       = lambda x, y: -y + 2.0 * np.where(np.asarray(x) >= 2.5, 1.0, 0.0),
    x_range = (0, 8),
    y_range = (-3, 7),
    density = 22,
    title   = "y' = -y + 2 H(x - 2.5)  (step source)",
)
```

#### Delta source

Because $\delta(x - x_0)$ is a distribution rather than a smooth function,
the vector field shows $y' = ay$ everywhere. The delta appears only in the
solution curve as a jump of size $q_0$ at $x_0$.

```python
plot_vector_field(
    F       = lambda x, y: -1.0 * y,
    x_range = (0, 8),
    y_range = (-3, 7),
    density = 22,
    title   = "y' = -y  (homogeneous field; delta kick at x0=2.5 shown in solution)",
)
```

#### Exponential source

$$y' = -y + 2\,e^{0.5x}, \qquad b = 0.5 \neq a = -1$$

$$y(x) = y_0 e^{-x} + \frac{2}{0.5-(-1)}\!\left(e^{0.5x} - e^{-x}\right) = y_0 e^{-x} + \tfrac{4}{3}\!\left(e^{0.5x} - e^{-x}\right)$$

```python
plot_vector_field(
    F       = lambda x, y: -y + 2.0 * np.exp(0.5 * np.asarray(x, float)),
    x_range = (0, 5),
    y_range = (-3, 20),
    density = 22,
    title   = "y' = -y + 2 exp(0.5x)  (exponential source)",
)
```

#### Resonance — $b = a$

When $b = a$, the source frequency matches the natural decay rate.
The solution grows as $x e^{ax}$:

$$y' = -y + 2\,e^{-x}, \quad b = a = -1 \implies y(x) = \left(y_0 + 2x\right)e^{-x}$$

```python
plot_vector_field(
    F       = lambda x, y: -y + 2.0 * np.exp(-np.asarray(x, float)),
    x_range = (0, 8),
    y_range = (-1, 6),
    density = 22,
    title   = "y' = -y + 2 exp(-x)  (resonance: b = a = -1)",
)
```

---

### Worked examples — custom source functions

Any callable `F(x, y)` that accepts NumPy arrays can be used.

#### Sinusoidal forcing

$$y' = -y + \sin(x) \qquad \text{particular solution: } y_p = \tfrac{1}{2}(\sin x - \cos x)$$

```python
plot_vector_field(
    F       = lambda x, y: -y + np.sin(x),
    x_range = (-2 * np.pi, 2 * np.pi),
    y_range = (-2, 2),
    title   = "y' = -y + sin(x)",
)
```

#### Piecewise source (multiple steps)

$$q(x) = \begin{cases} 0 & x < 2 \\ 1 & 2 \leq x < 5 \\ 2 & 5 \leq x < 8 \\ 0 & x \geq 8 \end{cases}$$

```python
def piecewise_q(x):
    x = np.asarray(x)
    return np.where(x < 2, 0,
           np.where(x < 5, 1.0,
           np.where(x < 8, 2.0, 0.0)))

plot_vector_field(
    F       = lambda x, y: -0.5 * y + piecewise_q(x),
    x_range = (0, 12),
    y_range = (-1, 6),
    title   = "y' = -0.5y + piecewise q(x)",
)
```

#### Unstable system — $a > 0$

The equilibrium $y^* = q_0 / |a|$ exists but is **unstable** when $a > 0$:

$$\frac{dy}{dx} = 0.5\,y - 3, \qquad y^* = 6 \text{ (unstable)}$$

```python
plot_vector_field(
    F       = lambda x, y: 0.5 * y - 3.0,
    x_range = (0, 6),
    y_range = (-2, 10),
    n_curves= 6,
    title   = "y' = 0.5y - 3  (unstable, a > 0)",
)
```

#### Gaussian pulse source

$$q(x) = A\,\exp\!\left(-\left(\frac{x - \mu}{\sigma}\right)^{\!2}\right), \qquad A = 4,\quad \mu = 3,\quad \sigma = 0.5$$

```python
def gaussian_q(x, mu=3.0, sigma=0.5, A=4.0):
    return A * np.exp(-((np.asarray(x) - mu) / sigma) ** 2)

plot_vector_field(
    F       = lambda x, y: -y + gaussian_q(x),
    x_range = (0, 8),
    y_range = (-1, 5),
    title   = "y' = -y + 4 exp(-((x-3)/0.5)^2)  (Gaussian pulse)",
)
```

#### Overlaying the exact analytical solution on a vector field

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
    n_curves= 0,        # suppress auto-generated curves
    title   = "Step source with exact analytical solution overlaid",
)

y_exact = solve_step(xs, a, y0, q0, x0)
ax.plot(xs, np.clip(y_exact, -2, 8),
        color='red', lw=2.2, ls='--',
        label=r'exact solution, $y_0 = 0.5$', zorder=5)
ax.legend(fontsize=10)
plt.show()
```

---

## Summary of solutions

| Source $q(x)$ | General solution $y(x)$ | Steady state ($a < 0$) |
|---|---|---|
| $q_0$ | $\left(y_0 + \dfrac{q_0}{a}\right)e^{ax} - \dfrac{q_0}{a}$ | $y^* = -\dfrac{q_0}{a}$ |
| $q_0\, H(x-x_0)$ | $y_0 e^{ax} + \dfrac{q_0}{a}\!\left(e^{a(x-x_0)}-1\right)$ for $x \geq x_0$ | $y^* = -\dfrac{q_0}{a}$ after $x_0$ |
| $q_0\,\delta(x-x_0)$ | $\left(y_0 e^{ax_0} + q_0\right)e^{a(x-x_0)}$ for $x \geq x_0$ | none — free decay resumes |
| $q_0\,e^{bx},\ b \neq a$ | $y_0 e^{ax} + \dfrac{q_0}{b-a}\!\left(e^{bx}-e^{ax}\right)$ | depends on signs of $a,\, b$ |
| $q_0\,e^{ax}$ (resonance) | $\left(y_0 + q_0\, x\right)e^{ax}$ | none — grows as $x\,e^{ax}$ |

---

## Quick reference — which function to call

| Goal | Call |
|---|---|
| Slope field for any $\dfrac{dy}{dx} = F(x,y)$ | `plot_vector_field(F, ...)` |
| Solution + field for $y' = ay + q_0$ | `plot_case('constant', ...)` |
| Solution + field for $y' = ay + q_0\,H(x-x_0)$ | `plot_case('step', ...)` |
| Solution + field for $y' = ay + q_0\,\delta(x-x_0)$ | `plot_case('delta', ...)` |
| Solution + field for $y' = ay + q_0\,e^{bx}$ | `plot_case('exp', ...)` |
| Compare all four sources side by side | `plot_all_cases(...)` |
| Get $y(x)$ values as a NumPy array | `solve_constant` / `solve_step` / `solve_delta` / `solve_exponential` |
| Vector field demos for all four linear cases | `python linear_ode.py --case vf_demo` |

---

## File layout

```
your_project/
├── vector_field.py    ← general slope-field plotter
├── linear_ode.py      ← linear ODE solver (auto-detects vector_field.py)
└── README.md
```

Both files are self-contained and independently importable. `linear_ode.py`
degrades gracefully when `vector_field.py` is absent — it displays the
solution panel only, without the side-by-side vector field panel.
