import numpy as np
import matplotlib.pyplot as plt


def u_step(u: np.ndarray, dt: float, dx: float, v: float, mur_bc: bool = True) -> np.ndarray:
    """
    Calculate the next time step of the 2D wave equation with a 2nd order finite difference in space and time.
    """
    _, ny, nx = u.shape
    u[im1] = u[i]  
    u[i] = u[ip1]
    alpha = v*dt/dx
    alpha2 = alpha**2

    # Calculate the new time step
    u[next, 1:ny-1, 1:nx-1]  = alpha2 * (
                    u[now, 0:ny-2, 1:nx-1]
                + u[now, 2:ny,   1:nx-1]
                + u[now, 1:ny-1, 0:nx-2]
                + u[now, 1:ny-1, 2:nx]
                - 4*u[now, 1:ny-1, 1:nx-1]) \
                + (2 * u[now, 1:ny-1, 1:nx-1] - 
                    u[old, 1:ny-1, 1:nx-1])

    if mur_bc:
        #Mur absorbing boundary conditions
        kappa = (1 - alpha) / (1 + alpha) 
        u[ip1, 0, k] = \
            (u[now, 1, k] - kappa * (u[ip1, 1, k] - u[now, 0, k]))
        u[ip1, ny-1, k] = \
            (u[now, ny-2, k] + kappa * (u[now, ny-1, k] - u[ip1, ny-2, k]))
        u[ip1, j, 0] = \
            (u[now, j, 1] - kappa * (u[ip1, j, 1] - u[now, j, 0]))
        u[ip1, j, nx-1] = \
            (u[now, j, nx-2] + kappa * (u[now, j, nx-1] - u[ip1, j, nx-2]))
    else:
        #Dirichlet boundary conditions
        u[ip1, 0, k] = 0
        u[ip1, ny-1, k] = 0
        u[ip1, j, 0] = 0
        u[ip1, j, nx-1] = 0 
    return u


A = 80 #amplitude
dt = .02 #s
dx = .1 #m
f = 1 #Hz
v=2 #m/s
omega = 2 * np.pi * f

nt = 500
nx = 200
ny = 200

#Index helpers
next = ip1 = np.array([0]) #u[0]
now = i = np.array([1]) #u[1]
old = im1 = np.array([2]) #u[2]

#j = 1:ny-1 (center)
j = np.arange(ny)
#j = 0:ny-2 (upper neighbour)
jm1 = np.arange(0,ny-1)
jm1 = np.insert(jm1, 0, 0)
#j = 2:ny (lower neighbour)
jp1 = np.arange(1,ny)
jp1 = np.insert(jp1, ny-1, ny-1)

#k = 1:nx-1 (center)
k = np.arange(nx)
#k = 0:nx-2 (left neighbour)
km1 = np.arange(0,nx-1)
km1 = np.insert(km1, 0, 0)
#k = 2:nx (right neighbour)
kp1 = np.arange(1,nx)
kp1 = np.insert(kp1, nx-1, nx-1)

print((v*dt)/dx)
t_vec = np.arange(0, dt*nt, dt)
source_signal = A * np.sin(t_vec * omega) * np.clip(np.sin(0.5 * t_vec * omega), 0, 1)
##plot source signal
#plt.plot(t_vec, source_signal)
#plt.show()

u = np.zeros((3, ny, nx))

fig, ax = plt.subplots()
plot_max = 0.15*A
ax.axis("off")
cmap = plt.cm.bwr
cmap.set_bad('gray')

o_mask = np.zeros_like(u[0],dtype=bool)
o_mask[40:80, 40:80] = True

for t_idx in range(nt):
    plt.cla()
    #inject source signal
    u[next, ny//2, nx//2] = source_signal[t_idx]
    u = u_step(u, dt, dx, v, mur_bc = True)
    #Apply obstacle
    u[ip1, 40:80, 40:80] = 0
    uplot = np.squeeze(u[ip1].copy())
    uplot[o_mask] = np.nan
    plt.imshow(uplot, cmap=cmap)
    plt.clim(-plot_max, plot_max)
    ax.set_aspect('equal')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.pause(0.001)

