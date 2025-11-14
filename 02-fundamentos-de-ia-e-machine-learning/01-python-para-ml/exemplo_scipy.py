from scipy.integrate import solve_ivp

def dydt(t, y):
   return -0.5 * y
solution = solve_ivp(dydt, [0, 10], [2])
print(solution.y)