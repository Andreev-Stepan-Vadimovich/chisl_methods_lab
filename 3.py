import math

# ВХОДНЫЕ ПАРАМЕТРЫ
a = 0.0
b = 2.0
c = 0.0
d = 1.0
m = 20
eps = 0.001

# ПОДЫНТЕГРАЛЬНАЯ ФУНКЦИЯ
def f(x, t):
    """f(x, t) = cos(t / (1 + x^2) + 0.001 * x)"""
    return math.cos(t / (1.0 + x**2) + 0.001 * x)

# МЕТОД СРЕДНИХ ПРЯМОУГОЛЬНИКОВ С УДВОЕНИЕМ ШАГОВ
def integrate_rectangles(f, a, b, t, eps):
    """
    Вычисляет интеграл методом средних прямоугольников,
    удваивая число шагов N, пока |S_2N - S_N| < eps.
    
    Возвращает: (значение интеграла, итоговое N)
    """
    N = 2                # начальное число шагов
    prev_S = None        # предыдущее значение суммы
    
    while True:
        h = (b - a) / N
        S = 0.0
        for i in range(N):
            x_mid = a + (i + 0.5) * h    # середина i-го отрезка
            S += f(x_mid, t)
        S *= h
        
        # Проверка критерия остановки
        if prev_S is not None and abs(S - prev_S) < eps:
            return S, N
        
        prev_S = S
        N *= 2

# ОСНОВНОЙ ЦИКЛ ПО t
tau = (d - c) / m   # шаг по t

print(f"Метод: средние прямоугольники с удвоением шагов")
print(f"f(x,t) = cos(t/(1+x^2) + 0.001*x)")
print(f"a = {a}, b = {b}, c = {c}, d = {d}, m = {m}, eps = {eps}")
print()
print(f"{'j':>3} | {'t_j':>8} | {'F(t_j)':>14} | {'N':>6}")
print("-" * 42)

for j in range(m + 1):
    t_j = c + j * tau
    F_tj, N_used = integrate_rectangles(f, a, b, t_j, eps)
    print(f"{j:>3} | {t_j:>8.4f} | {F_tj:>14.8f} | {N_used:>6}")

print("-" * 42)