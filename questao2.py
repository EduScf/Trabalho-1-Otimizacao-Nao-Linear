import numpy as np
from scipy.optimize import minimize

# Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função-objetivo
def funcaoobjetivo(w):
    
    # Carregar os dados do problema
    dados = np.load('questao2_dados.npz')
    X = dados['X'] # Entrada: matriz de dimensões m x n
    y = dados['y'] # Saída: rótulos binários
    m, n = X.shape
    lambd = 0.1

    h = sigmoid(X @ w)
    loss = -np.mean(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))
    reg = (lambd / 2) * np.sum(w**2)
    return loss + reg


# Chute inicial dos pesos
n = 500
w0 = np.zeros(n)

""" IMPLEMENTE AQUI A CHAMADA DO ALGORTIMO DE OTIMIZAÇÃO """

# Gradiente da função-objetivo
def gradiente(w):
    dados = np.load('questao2_dados.npz')
    X = dados['X']
    y = dados['y']
    m = X.shape[0]
    lambd = 0.1

    h = sigmoid(X @ w)
    grad = (1 / m) * (X.T @ (h - y)) + lambd * w
    return grad

# Otimização com gradientes conjugados (Fletcher-Reeves)
resultado = minimize(
    fun=funcaoobjetivo,
    x0=w0,
    method='CG',
    jac=gradiente,
    options={'disp': True, 'maxiter': 1000}
)

# Pesos otimizados
custo_final = resultado.fun
numero_iteracoes = resultado.nit
numero_avaliacoes = resultado.nfev
print(f"Custo final: {custo_final:.4f}")
print(f"Número de iterações: {numero_iteracoes}")
print(f"Número de avaliações da função-objetivo: {numero_avaliacoes}")