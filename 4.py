import numpy as np
import matplotlib.pyplot as plt

def system(x, y):
    """
    dx/dt = (2x - y)^2 - 9
    dy/dt = 9 - (x - 2y)^2
    """
    dx = (2*x - y)**2 - 9
    dy = 9 - (x - 2*y)**2
    return dx, dy

def rk4_step(x, y, h):
    f1x, f1y = system(x, y)
    Q1x, Q1y = h * f1x, h * f1y

    f2x, f2y = system(x + Q1x / 3, y + Q1y / 3)
    Q2x, Q2y = h * f2x, h * f2y

    f3x, f3y = system(x - Q1x / 3 + Q2x, y - Q1y / 3 + Q2y)
    Q3x, Q3y = h * f3x, h * f3y

    f4x, f4y = system(x + Q1x - Q2x + Q3x, y + Q1y - Q2y + Q3y)
    Q4x, Q4y = h * f4x, h * f4y

    x_new = x + (Q1x + 3 * Q2x + 3 * Q3x + Q4x) / 8
    y_new = y + (Q1y + 3 * Q2y + 3 * Q3y + Q4y) / 8
    return x_new, y_new

def integrate_trajectory(x0, y0, h=0.05, t_max=5.0, lim=100.0,
                         center=None, radius=None):
    n_steps = int(round(t_max / h))
    xs, ys, ts = [x0], [y0], [0.0]
    x, y = x0, y0

    for i in range(n_steps):
        x, y = rk4_step(x, y, h)
        if not np.isfinite(x) or not np.isfinite(y) or abs(x) > lim or abs(y) > lim:
            break
        if center is not None and radius is not None:
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2:
                break
        xs.append(x)
        ys.append(y)
        ts.append((i + 1) * h)

    return np.array(xs), np.array(ys), np.array(ts)

def jacobian(x, y):
    a = 4*(2*x - y)
    b = -2*(2*x - y)
    c = -2*(x - 2*y)
    d = 4*(x - 2*y)
    return np.array([[a, b], [c, d]])

def classify_point(x0, y0):
    J = jacobian(x0, y0)
    eigvals = np.linalg.eigvals(J)
    re, im = eigvals.real, eigvals.imag

    if abs(im[0]) > 1e-9:
        if abs(re[0]) < 1e-9:
            return "Центр", eigvals
        elif re[0] > 0:
            return "Неустойчивый фокус", eigvals
        else:
            return "Устойчивый фокус", eigvals
    else:
        if re[0] > 0 and re[1] > 0:
            return "Неустойчивый узел", eigvals
        elif re[0] < 0 and re[1] < 0:
            return "Устойчивый узел", eigvals
        elif re[0] * re[1] < 0:
            return "Седло", eigvals
        else:
            return "Вырожденный случай", eigvals

critical_points = [(1, -1), (3, 3), (-3, -3), (-1, 1)]

print("=" * 70)
print("АНАЛИЗ ОСОБЫХ ТОЧЕК")
print("=" * 70)
kinds = {}
for x0, y0 in critical_points:
    kind, eigvals = classify_point(x0, y0)
    kinds[(x0, y0)] = kind
    print(f"Точка ({x0:>3}, {y0:>3}): {kind:>25}  "
          f"λ = {eigvals[0].real:.4f}, {eigvals[1].real:.4f}")
print("=" * 70)


h = 0.05
h_plot = 0.01
eps = 0.2
radius = 0.5

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (x0, y0) in enumerate(critical_points):
    ax = axes[idx]

    starts = [
        (x0 + eps, y0, '#d46a8c', f'({x0+eps:.2f}, {y0:.2f})'),
        (x0 - eps, y0, '#6a5acd', f'({x0-eps:.2f}, {y0:.2f})'),
        (x0, y0 + eps, '#3aa0c8', f'({x0:.2f}, {y0+eps:.2f})'),
        (x0, y0 - eps, '#8a6bb5', f'({x0:.2f}, {y0-eps:.2f})'),
    ]

    for sx, sy, col, label in starts:
        xs, ys, ts = integrate_trajectory(
            sx, sy, h=h_plot, t_max=3.0, lim=50.0,
            center=(x0, y0), radius=radius,
        )
        ax.plot(xs, ys, color=col, linewidth=1.8, label=label)
        ax.plot(xs[0], ys[0], 'o', color=col, markersize=7,
                markeredgecolor='black', markeredgewidth=0.6)

    ax.plot(x0, y0, marker='*', color='#3d2a7a', markersize=16,
            markeredgecolor='black', markeredgewidth=0.4, label='Особая точка', zorder=5)

    ax.set_title(f'({x0}, {y0}) — {kinds[(x0, y0)]}', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x0 - 0.55, x0 + 0.55)
    ax.set_ylim(y0 - 0.55, y0 + 0.55)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc='best')

plt.tight_layout()
plt.savefig('phase_portraits.png', dpi=120, bbox_inches='tight')
plt.show()

print("\nУстойчивый узел, старт (-0.8, 1.0), h = 0.05:")
print(f"{'t':>6} | {'x':>12} | {'y':>12}")
print("-" * 36)
xs, ys, ts = integrate_trajectory(-0.8, 1.0, h=h, t_max=1.0, lim=50.0)
for i in range(0, len(ts), 2):
    print(f"{ts[i]:>6.2f} | {xs[i]:>12.6f} | {ys[i]:>12.6f}")

print("\nНеустойчивый узел, старт (1.2, -1.0), h = 0.05:")
print(f"{'t':>6} | {'x':>12} | {'y':>12}")
print("-" * 36)
xs, ys, ts = integrate_trajectory(1.2, -1.0, h=h, t_max=1.0, lim=30.0)
for i in range(len(ts)):
    print(f"{ts[i]:>6.2f} | {xs[i]:>12.6f} | {ys[i]:>12.6f}")
