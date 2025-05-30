import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Função que representa o sistema de controle (sistema de primeira ordem)
def system(T, t, u):
    # Equação diferencial de um sistema de primeira ordem
    # dT/dt = -T + u
    return -T + u

# Função que define o controlador PID
def pid_controller(Kp, Ki, Kd, e, e_integral, e_derivative):
    return Kp * e + Ki * e_integral + Kd * e_derivative

# Função para simular o sistema controlado
def simulate_pid(Kp, Ki, Kd, T_ref, T0, t):
    T = T0  # Temperatura inicial do sistema
    e_integral = 0  # Parte integral do erro
    e_previous = 0  # Erro anterior (para derivada)
    response = []  # Armazenar a resposta do sistema

    for i in range(len(t)):
        # Erro entre a referência e o valor atual
        e = T_ref - T
        # Integral do erro
        e_integral += e * (t[1] - t[0])
        # Derivada do erro
        e_derivative = (e - e_previous) / (t[1] - t[0])

        # Sinal de controle (PID)
        u = pid_controller(Kp, Ki, Kd, e, e_integral, e_derivative)

        # Atualizar a temperatura usando a equação diferencial
        T = odeint(system, T, [t[i], t[i] + (t[1] - t[0])], args=(u,))[-1]

        # Armazenar a resposta e atualizar o erro anterior
        response.append(T)
        e_previous = e

    return np.array(response)

# Função-objetivo: calcular o erro quadrático médio (MSE) entre a resposta e a referência
def funcaoobjetivo(x):
    
    # Parâmetros de simulação
    T_ref = 100.0   # Temperatura de referência (setpoint)
    T0 = 25.0      # Temperatura inicial do sistema
    t = np.linspace(0, 10, 100)  # Tempo de simulação

    Kp, Ki, Kd = x
    response = simulate_pid(Kp, Ki, Kd, T_ref, T0, t)
    mse = np.mean((T_ref - response) ** 2)
    return mse


# Chute inicial para os parâmetros do controlador PID
ponto_inicial = [1.0, 0.1, 0.01]

""" IMPLEMENTE AQUI A CHAMADA DO ALGORTIMO DE OTIMIZAÇÃO """

# Extraia aqui os parâmetros PID otimizados
Kp_opt, Ki_opt, Kd_opt = ##

# Simulação final com os parâmetros otimizados
response_opt = simulate_pid(Kp_opt, Ki_opt, Kd_opt)

# Plotar a resposta do sistema
T_ref = 100.0
t = np.linspace(0, 10, 100)
plt.plot(t, response_opt, label=f'Otimizado: Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f}')
plt.axhline(y=T_ref, color='r', linestyle='--', label='Referência')
plt.xlabel('Tempo')
plt.ylabel('Temperatura')
plt.legend()
plt.title('Resposta do Sistema Controlado (PID Otimizado)')
plt.show()

# Exibir os parâmetros otimizados
print(f"Parâmetros PID otimizados: Kp={Kp_opt:.2f}, Ki={Ki_opt:.2f}, Kd={Kd_opt:.2f}")
