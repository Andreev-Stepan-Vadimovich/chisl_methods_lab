import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. СИСТЕМА ДИФФЕРЕНЦИАЛЬНЫХ УРАВНЕНИЙ (ВАРИАНТ 13)
# ==========================================
def system(x, y):
    """
    dx/dt = (2x - y)^2 - 9
    dy/dt = 9 - (x - 2y)^2
    """
    dx = (2*x - y)**2 - 9
    dy = 9 - (x - 2*y)**2
    return dx, dy

# ==========================================
# 2. МЕТОД РУНГЕ-КУТТЫ 4-ГО ПОРЯДКА (МЕТОД №1)
# ==========================================
def rk4_step(x, y, h):
    """
    Один шаг RK4 для системы из двух уравнений.
    Формулы из методички (стр. 51, метод 1):
        Q1 = h * f(x_k, y_k)
        Q2 = h * f(x_k + h/2, y_k + Q1/2)
        Q3 = h * f(x_k + h/2, y_k + Q2/2)
        Q4 = h * f(x_k + h, y_k + Q3)
        y_{k+1} = y_k + (Q1 + 2Q2 + 2Q3 + Q4) / 6
    """
    Q1x, Q1y = system(x, y)
    Q1x *= h
    Q1y *= h
    
    Q2x, Q2y = system(x + h/2, y + Q1y/2)
    Q2x *= h
    Q2y *= h
    
    Q3x, Q3y = system(x + h/2, y + Q2y/2)
    Q3x *= h
    Q3y *= h
    
    Q4x, Q4y = system(x + h, y + Q3y)
    Q4x *= h
    Q4y *= h
    
    x_new = x + (Q1x + 2*Q2x + 2*Q3x + Q4x) / 6
    y_new = y + (Q1y + 2*Q2y + 2*Q3y + Q4y) / 6
    return x_new, y_new

def integrate_trajectory(x0, y0, h=0.05, t_max=5.0, lim=100.0):
    """
    Интегрирует траекторию от (x0, y0) до t_max.
    Останавливается, если |x| или |y| выходит за пределы [-lim, lim]
    (защита от OverflowError и от бесконечного роста).
    
    Возвращает: xs, ys, t_end (реальное время до остановки)
    """
    n_steps = int(t_max / h)
    xs, ys, ts = [x0], [y0], [0.0]
    x, y = x0, y0
    
    for i in range(n_steps):
        x, y = rk4_step(x, y, h)
        
        # Защита от переполнения / ухода на бесконечность
        if not np.isfinite(x) or not np.isfinite(y) or abs(x) > lim or abs(y) > lim:
            break
        
        xs.append(x)
        ys.append(y)
        ts.append((i + 1) * h)
    
    return np.array(xs), np.array(ys), np.array(ts)

# ==========================================
# 3. АНАЛИЗ ОСОБЫХ ТОЧЕК
# ==========================================
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
print("АНАЛИЗ ОСОБЫХ ТОЧЕК (вариант 13)")
print("=" * 70)
for x0, y0 in critical_points:
    kind, eigvals = classify_point(x0, y0)
    print(f"Точка ({x0:>3}, {y0:>3}): {kind:>25}  "
          f"λ = {eigvals[0].real:.4f}, {eigvals[1].real:.4f}")
print("=" * 70)

# ==========================================
# 4. ПОСТРОЕНИЕ ГРАФИКОВ ПО ОСОБЫМ ТОЧКАМ
# ==========================================
h = 0.05
t_max = 5.0
eps = 0.3

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (x0, y0) in enumerate(critical_points):
    ax = axes[idx]
    
    starts = [
        (x0 + eps, y0, 'red',    f'({x0+eps:.2f}, {y0:.2f})'),
        (x0 - eps, y0, 'blue',   f'({x0-eps:.2f}, {y0:.2f})'),
        (x0, y0 + eps, 'green',  f'({x0:.2f}, {y0+eps:.2f})'),
        (x0, y0 - eps, 'purple', f'({x0:.2f}, {y0-eps:.2f})'),
    ]
    
    for sx, sy, col, label in starts:
        xs, ys, ts = integrate_trajectory(sx, sy, h=h, t_max=t_max, lim=50.0)
        
        ax.plot(xs, ys, color=col, linewidth=1.5, label=f'старт {label}')
        ax.plot(xs[0], ys[0], 'o', color=col,
    markersize=8, markeredgecolor='black')
        ax.plot(xs[-1], ys[-1], '*', color=col, markersize=14, markeredgecolor='black')
    
    # Особая точка
    ax.plot(x0, y0, 'kX', markersize=15, markeredgewidth=2, label='Особая точка')
    
    ax.set_title(f'Точка ({x0}, {y0})', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    # Не устанавливаем equal aspect — траектории могут уходить далеко

plt.suptitle('Фазовые траектории в окрестности особых точек (вариант 13)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('phase_portraits.png', dpi=120, bbox_inches='tight')
plt.show()

# ==========================================
# 5. ОБЩИЙ ФАЗОВЫЙ ПОРТРЕТ
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10))

# Векторное поле
X, Y = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(-6, 6, 25))
U, V = system(X, Y)
norm = np.sqrt(U**2 + V**2) + 1e-9
ax.quiver(X, Y, U/norm, V/norm, norm, cmap='viridis', alpha=0.6, scale=25)

# Траектории вокруг каждой особой точки
for x0, y0 in critical_points:
    for sx, sy in [(x0+eps, y0), (x0-eps, y0), (x0, y0+eps), (x0, y0-eps)]:
        xs, ys, ts = integrate_trajectory(sx, sy, h=h, t_max=t_max, lim=50.0)
        ax.plot(xs, ys, color='red', linewidth=1, alpha=0.7)
    ax.plot(x0, y0, 'kX', markersize=15, markeredgewidth=2)

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Общий фазовый портрет системы (вариант 13)\n'
             '× — особые точки', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('general_phase.png', dpi=120, bbox_inches='tight')
plt.show()

# ==========================================
# 6. ТАБЛИЦА РЕШЕНИЯ ДЛЯ ОДНОЙ ТРАЕКТОРИИ
# ==========================================
print("\nПример численного решения (старт из точки (1.3, -1.0)):")
print(f"{'t':>6} | {'x':>12} | {'y':>12}")
print("-" * 36)

xs, ys, ts = integrate_trajectory(1.3, -1.0, h=h, t_max=1.0, lim=50.0)
for i in range(0, len(ts), 4):   # каждый 4-й шаг для компактности
    print(f"{ts[i]:>6.2f} | {xs[i]:>12.6f} | {ys[i]:>12.6f}")