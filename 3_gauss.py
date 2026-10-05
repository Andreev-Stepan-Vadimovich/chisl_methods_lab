import math

a = 0.0
b = 2.0
c = 0.0
d = 1.0
m = 20

def f(x, t):
    return math.cos(t / (1.0 + x**2) + 0.001 * x)

# Узлы и веса Гаусса на отрезке [-1, 1].
# Для трёх узлов — точные значения из методички.
# Для четырёх — табличные, как напечатаны в задании.
X3 = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
C3 = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

X4 = [-0.861136, -0.339981, 0.339981, 0.861136]
C4 = [0.347855, 0.652145, 0.652145, 0.347855]


def gauss(f, a, b, t, nodes, weights):
    """
    Квадратура Гаусса на [a, b].

    Узлы и веса заданы для [-1, 1]. На [a, b] они переносятся так:
        u = (b - a) / 2 * x + (b + a) / 2
        d = (b - a) / 2 * c
    """
    half = (b - a) / 2.0
    mid = (b + a) / 2.0
    S = 0.0
    for x, c_i in zip(nodes, weights):
        u = half * x + mid
        d = half * c_i
        S += d * f(u, t)
    return S


tau = (d - c) / m

print("=" * 62)
print("Метод: квадратуры Гаусса с 3 и 4 узлами")
print("=" * 62)
print("f(x, t) = cos(t/(1+x^2) + 0.001*x)")
print(f"a={a}, b={b}, c={c}, d={d}, m={m}")
print("=" * 62)
print(f"{'j':>3} | {'t_j':>8} | {'F3(t_j)':>16} | {'F4(t_j)':>16}")
print("-" * 62)

for j in range(m + 1):
    t_j = c + j * tau
    F3 = gauss(f, a, b, t_j, X3, C3)
    F4 = gauss(f, a, b, t_j, X4, C4)
    print(f"{j:>3} | {t_j:>8.4f} | {F3:>16.10f} | {F4:>16.10f}")

print("=" * 62)
