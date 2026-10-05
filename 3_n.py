import math

a = 0.0
b = 2.0
c = 0.0
d = 1.0
m = 20
eps = 0.001

def f(x, t):
    # return math.exp(-(1.0 / (1.0 + x**2) + 0.001 * x))
    return math.exp(-(t / (1.0 + x**2) + 0.001 * x))

def integrate_simpson(f, a, b, t, eps):
    N = 2
    prev_S = None
    
    while True:
        h = (b - a) / N
        S = f(a, t) + f(b, t)
        
        for i in range(1, N):
            x_i = a + i * h
            if i % 2 == 1:
                S += 4.0 * f(x_i, t)  
            else:
                S += 2.0 * f(x_i, t) 
        
        S *= h / 3.0
        
        if prev_S is not None and abs(S - prev_S) < eps:
            return S, N
        
        prev_S = S
        N *= 2                    

tau = (d - c) / m

print("=" * 45)
print("Метод: Симпсона с удвоением числа шагов")
print("=" * 45)
print(f"f(x, t) = exp(-(t/(1+x^2) + 0.001x))")
print(f"a={a}, b={b}, c={c}, d={d}, m={m}, eps={eps}")
print("=" * 45)
print(f"{'j':>3} | {'t_j':>8} | {'F(t_j)':>16} | {'N':>6}")
print("-" * 45)

for j in range(m + 1):
    t_j = c + j * tau
    F_tj, N_used = integrate_simpson(f, a, b, t_j, eps)
    print(f"{j:>3} | {t_j:>8.4f} | {F_tj:>16.10f} | {N_used:>6}")

print("=" * 45)