import numpy as np
import matplotlib.pyplot as plt

# Função de custo negativa (para maximização do lucro)
def funcaoobjetivo(x):
    d, t, m = x  # Desconto, tempo, orçamento
    VB = 100000  # Vendas básicas
    CB = 10000  # Custo fixo inicial
    
    # Receita
    f1 = -0.005 * d**2 + 0.2 * d
    f2 = 0.05 * t
    receita = VB * (1 + f1 + f2) * np.log(1 + m)
    
    # Custo
    custo_marketing = m
    penalidades = 0
    
    """
    Implemente agora as penalidades. Por exemplo:
    
    if x > 100:
        penalidades += 5000
    """

    if d < 0 or d > 50 or t < 1 or t > 30 or m < 1000 or m > 50000:
        return 1e6  # Penalidade alta para valores fora do intervalo
    
    if d > 30:
        penalidades += 5000
    if t > 15:
        penalidades += 2000
    
    custo_total =   CB + custo_marketing + penalidades
    
    # Lucro
    lucro = receita - custo_total
    
    # Lembre-se que eu quero maximizar o lucro e meu algoritmo de otimização minimiza a função objetivo
    return -lucro  

# Chute inicial
ponto_inicial = [10, 10, 10000]

""" IMPLEMENTE AQUI A CHAMADA DO ALGORTIMO DE OTIMIZAÇÃO """

def hooke_jeeves(func, x0, step=1.0, alpha=2.0, epsilon=1e-3):
    n = len(x0)
    x = np.array(x0)
    progresso = [-func(x)]  # registra lucro inicial

    while step > epsilon:
        y = np.copy(x)
        for i in range(n):
            xp = np.copy(y)
            xp[i] += step
            if func(xp) < func(y):
                y = xp
            else:
                xm = np.copy(y)
                xm[i] -= step
                if func(xm) < func(y):
                    y = xm

        if func(y) < func(x):
            z = y + alpha * (y - x)
            if func(z) < func(y):
                x = z
            else:
                x = y
        else:
            step /= 2.0
        progresso.append(-func(x))  # salva lucro atual

    return x, -func(x), progresso

# Chute inicial
x0 = [10, 10, 10000]  # d, t, m

# Executar o algoritmo
x_opt, lucro_max, progresso = hooke_jeeves(funcaoobjetivo, x0)

# Resultados
d, t, m = x_opt

print(f"Parâmetros ótimos: Desconto = {d:.2f}%, Tempo = {t:.2f} dias, Orçamento = ${m:.2f}")
print(f"Lucro máximo estimado: R${lucro_max:.2f}")

# Plot do progresso
plt.figure(figsize=(10, 5))
plt.plot(progresso, marker='o')
plt.xlabel('Iteração')
plt.ylabel('Lucro (R$)')
plt.title('Evolução do Lucro durante o Hooke-Jeeves')
plt.grid(True)
plt.tight_layout()
plt.show()