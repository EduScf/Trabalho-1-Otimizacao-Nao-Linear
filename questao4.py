import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Função objetivo (da aleta)
def f(x):
    x1, x2 = x
    return (
        0.6382 * x1**2 +
        0.3191 * x2**2 -
        0.2809 * x1 * x2 -
        67.906 * x1 -
        14.29 * x2
    )

# Gradiente analítico
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2 * 0.6382 * x1 - 0.2809 * x2 - 67.906
    df_dx2 = 2 * 0.3191 * x2 - 0.2809 * x1 - 14.29
    return np.array([df_dx1, df_dx2])

# Ponto inicial razoável
x0 = np.array([50.0, 20.0], dtype=np.float64)

# Otimização com BFGS
res = minimize(
    f, x0,
    method='BFGS',
    jac=grad_f,
    options={'gtol': 1e-8, 'disp': True, 'eps': 1e-4}
)

# Resultados
x1_opt, x2_opt = res.x
print(f"Solução ótima: x1 = {x1_opt:.4f}, x2 = {x2_opt:.4f}")
print(f"Valor mínimo da função: f(x) = {res.fun:.4f}")
