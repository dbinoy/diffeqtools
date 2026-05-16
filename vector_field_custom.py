from vector_field import plot_vector_field
import numpy as np

# dy/dx = y − x²
plot_vector_field(
    # F       = lambda x, y: y - x**2,
    F       = lambda x, y: 3+6*y,
    x_range = (-3, 3),
    y_range = (-3, 5),
    density = 22,
    # title   = "dy/dx = y − x²",
    title   = "dy/dx = 3+6y",
)