import numpy as np

# Função objetivo (problema da aleta)
def f(x):
    x1, x2 = x
    return (
        0.6382 * x1**2 +
        0.3191 * x2**2 -
        0.2809 * x1 * x2 -
        67.906 * x1 -
        14.29 * x2
    )

# Gradiente da função
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2 * 0.6382 * x1 - 0.2809 * x2 - 67.906
    df_dx2 = 2 * 0.3191 * x2 - 0.2809 * x1 - 14.29
    return np.array([df_dx1, df_dx2])

# Hessiana constante (porque a função é quadrática)
def hessiana():
    return np.array([
        [2 * 0.6382, -0.2809],
        [-0.2809, 2 * 0.3191]
    ])

# Método de Newton manual
def metodo_newton(f, grad_f, hessiana, x0, epsilon=1e-8, max_iter=10):
    x = x0.copy()
    H = hessiana()
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < epsilon:
            break
        delta = np.linalg.solve(H, g)
        x = x - delta
    return x, f(x), k  # número real de passos realizados

# Execução
x0 = np.array([1.0, 1.0])  # ponto inicial genérico
x_opt, f_min, iteracoes = metodo_newton(f, grad_f, hessiana, x0)

# Resultados
print(f"Solução ótima: x1 = {x_opt[0]:.4f}, x2 = {x_opt[1]:.4f}")
print(f"Valor mínimo da função: f(x) = {f_min:.4f}")
print(f"Número de iterações (passos de Newton): {iteracoes}")
