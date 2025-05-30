import numpy as np

# Função de custo negativa (para maximização do lucro)
def funcaoobjetivo(x):
    d, t, m = x  # Desconto, tempo, orçamento
    VB = 100000  # Vendas básicas
    CB = 10000  # Custo fixo inicial
    
    # Receita
    f1 = ???
    f2 = ???
    receita = ???
    
    # Custo
    custo_marketing = m
    penalidades = 0
    
    """
    Implemente agora as penalidades. Por exemplo:
    
    if x > 100:
        penalidades += 5000
    """
    
    custo_total = ???
    
    # Lucro
    lucro = receita - custo_total
    
    # Lembre-se que eu quero maximizar o lucro e meu algoritmo de otimização minimiza a função objetivo
    return ???  

# Chute inicial
ponto_inicial = [10, 10, 10000]

""" IMPLEMENTE AQUI A CHAMADA DO ALGORTIMO DE OTIMIZAÇÃO """

# Resultados
d = ???
t = ???
m = ???
lucro = ???

print(f"Parâmetros ótimos: Desconto = {d:.2f}%, Tempo = {t:.2f} dias, Orçamento = ${m:.2f}")
print(f"Lucro máximo estimado: R${lucro:.2f}")