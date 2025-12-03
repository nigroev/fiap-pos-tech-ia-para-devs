# Exemplo: resolver uma equação diferencial
# Caso de uso: Modelagem de sistemas físicos, como o movimento de objetos em engenharia.

from scipy.integrate import solve_ivp

def dydt(t, y):
   return -0.5 * y
solution = solve_ivp(dydt, [0, 10], [2])
print(solution.y)