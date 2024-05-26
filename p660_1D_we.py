import numpy as np
import matplotlib.pyplot as plt

# Parameters
v = 1.0             # Wave speed
dx = 0.01           # Space step
dt = 0.004          # Time step
T_max = 2.0         # Maximum time
boundary = "Dirichlet"  # "Dirichlet" or "Neumann"

print("CFL ", (v*dt)/dx)

# Initialization
x = np.arange(0, 1+dx, dx)
u = np.zeros_like(x)  # u at time n
u_new = np.zeros_like(x)  # u at time n+1
u_old = np.zeros_like(x)  # u at time n-1

# Initial condition
n = 4
u_init = np.sin(n * np.pi * x)

# Set the initial condition u_old for the first time step
u = u_init.copy()
u_old = u.copy()

fig, ax = plt.subplots()

# Time-stepping loop
for t in np.arange(0, T_max, dt):
    plt.cla()
    # Update all points along the string
    for i in range(1, len(x) - 1):
        u_new[i] = (v * dt / dx)**2 * (u[i + 1] - 2 * u[i] +
                                       u[i - 1]) + 2 * u[i] - u_old[i]

    # Apply boundary conditions
    if boundary == "Dirichlet":
        u_new[0] = u_new[-1] = 0.0  # Fixed ends
    elif boundary == "Neumann":
        u_new[0] = u_new[1]  # Derivative at the end is zero (loose end)
        u_new[-1] = u_new[-2]

    # Update u_old and u
    u_old = u.copy()
    u = u_new.copy()

    # Plot the final state of the string
    plt.plot(x, u, 'k')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim([-1.1 * np.max(np.abs(u_init)), 1.1 * np.max(np.abs(u_init))])
    plt.pause(0.0001)
