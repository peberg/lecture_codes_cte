import time
import numpy as np
import matplotlib.pyplot as plt
from p200_cte_waves import source_signal, index_helpers


def fd_step_varden(u: np.ndarray,
                   dt: float,
                   dx: float,
                   v: np.ndarray,
                   rho: np.ndarray,
                   method: str = 1,
                   mur_bc: bool = True,
                   mur_rim_width: int = 3) -> np.ndarray:
    """
    Calculate the next time step of the 2D wave equation with a 2nd order finite difference in space and time.

    u: np.ndarray
        3D array with shape (3, ny, nx) containing the wave field at three consecutive time steps.
        u[0] is the wave field at the next time step.
        u[1] is the wave field at the current time step.
        u[2] is the wave field at the previous time step.
    """
    _, ny, nx = u.shape
    alpha = v * dt / dx
    alpha2 = alpha**2

    #used for 'index_helpers' and 'mur_bc'
    ip1, i, im1, jp1, j, jm1, kp1, k, km1 = index_helpers(u)
    u[2] = u[1]  #now -> old
    u[1] = u[0]  #next -> now

    if method == 1:
        #file:///home/bergmann/Downloads/applsci-13-08852.pdf
        #Simulation of Wave Propagation Using Finite Differences in Oil Exploration
        #variable density
        #        for (j,k),_ in np.ndenumerate(u[0]):
        #            if j == 0 or j == ny-1 or k == 0 or k == nx-1:
        #                pass
        #            else:
        #                u[0, j, k] = 2 * u[1, j, k] - u[2, j, k] + \
        #                        (rho[j, k] * v[j, k]**2 * dt**2)/(2*dx**2) * \
        #                        ((1/rho[j+1, k] + 1/rho[j, k]) * (u[1, j+1, k] - u[1, j, k]) -
        #                        (1/rho[j, k] + 1/rho[j-1, k]) * (u[1, j, k] - u[1, j-1, k]) +
        #                        (1/rho[j, k+1] + 1/rho[j, k]) * (u[1, j, k+1] - u[1, j, k]) -
        #                        (1/rho[j, k] + 1/rho[j, k-1]) * (u[1, j, k] - u[1, j, k-1]))

        u[ip1, j[:,None], k] = 2 * u[i, j[:,None], k] - u[im1, j[:,None], k] + \
                (rho[j[:,None], k] * v[j[:,None], k]**2 * dt**2)/(2*dx**2) * \
                ((1/rho[jp1[:,None], k] + 1/rho[j[:,None], k]) * (u[i, jp1[:,None], k] - u[i, j[:,None], k]) -
                    (1/rho[j[:,None], k] + 1/rho[jm1[:,None], k]) * (u[i, j[:,None], k] - u[i, jm1[:,None], k]) +
                    (1/rho[j[:,None], kp1] + 1/rho[j[:,None], k]) * (u[i, j[:,None], kp1] - u[i, j[:,None], k]) -
                    (1/rho[j[:,None], k] + 1/rho[j[:,None], km1]) * (u[i, j[:,None], k] - u[i, j[:,None], km1]))

    elif method == 2:
        for (j, k), _ in np.ndenumerate(u[0]):
            if j == 0 or j == ny - 1 or k == 0 or k == nx - 1:
                pass
            else:
                u[0, j, k] = (2 * u[1, j, k] - u[2, j, k] +
                              (v[j, k]**2 * dt**2 / (2 * dx**2)) *
                              (((u[2, j + 1, k] - u[2, j, k]) *
                                (rho[j + 1, k] + rho[j, k]) / rho[j + 1, k]) -
                               ((u[2, j, k] - u[2, j - 1, k]) *
                                (rho[j, k] + rho[j - 1, k]) / rho[j - 1, k]) +
                               ((u[2, j, k + 1] - u[2, j, k]) *
                                (rho[j, k + 1] + rho[j, k]) / rho[j, k + 1]) -
                               ((u[2, j, k] - u[2, j, k - 1]) *
                                (rho[j, k] + rho[j, k - 1]) / rho[j, k - 1])))

    if mur_bc:
        #Mur absorbing boundary conditions
        kappa = dt * v / dx
        top, bot = ny - 1, 0
        right, left = nx - 1, 0
        rim = mur_rim_width

        u[0, ny - rim - 1:top, 1:nx -
          1] = u[1, ny - rim - 2:top - 1, 1:nx -
                 1] + (kappa[ny - rim - 1:top, 1:nx - 1] -
                       1) / (kappa[ny - rim - 1:top, 1:nx - 1] +
                             1) * (u[0, ny - rim - 2:top - 1, 1:nx - 1] -
                                   u[1, ny - rim - 1:top, 1:nx - 1])

        u[0, bot:rim, 1:nx - 1] = u[1, bot + 1:rim + 1, 1:nx - 1] + (
            kappa[bot:rim, 1:nx - 1] - 1) / (kappa[bot:rim, 1:nx - 1] + 1) * (
                u[0, bot + 1:rim + 1, 1:nx - 1] - u[1, bot:rim, 1:nx - 1])

        u[0, 1:ny - 1, nx - 1 -
          rim:right] = u[1, 1:ny - 1, nx - 2 - rim:right -
                         1] + (kappa[1:ny - 1, nx - 1 - rim:right] - 1) / (
                             kappa[1:ny - 1, nx - 1 - rim:right] +
                             1) * (u[0, 1:ny - 1, nx - 2 - rim:right - 1] -
                                   u[1, 1:ny - 1, nx - 1 - rim:right])

        u[0, 1:ny - 1, left:rim] = u[1, 1:ny - 1, left + 1:rim + 1] + (kappa[
            1:ny - 1, left:rim] - 1) / (kappa[1:ny - 1, left:rim] + 1) * (
                u[0, 1:ny - 1, left + 1:rim + 1] - u[1, 1:ny - 1, left:rim])

    else:
        #Dirichlet boundary conditions
        u[ip1, 0, k] = 0
        u[ip1, ny - 1, k] = 0
        u[ip1, j, 0] = 0
        u[ip1, j, nx - 1] = 0
    return u


A = 80 #amplitude
f = 150 #Hz
omega = 2 * np.pi * f

dt = .0001 #s
nt = 1200
dx = .05 #m
nx, ny = 600, 300

#Create velocity and density fields
v = 330 * np.ones((ny, nx)) #m/s
v[:int(0.5*ny),:] = 220
rho = np.ones((ny, nx)) #kg/m^3
rho[:int(0.5*ny),:] = 2

#Create time vector
t_vec = np.arange(0, dt*nt, dt)
method = 'numpy_indexing'
#'numpy_indexing' or 'index_helpers'
print('CFL: ',np.max((v*dt)/dx))

#Create source signal
signal = source_signal(t_vec, A, f,
    source_type = 'pulsed_sine', plot = False)

#signal = source_bandlimited_random(t_vec, A, 100, 200, plot = True)
fig, ax = plt.subplots()
ax.axis("off")
plot_max = 0.12*A
cmap = plt.cm.bwr
cmap.set_bad('gray')

#Initialize wave field array and set obstacles
u = np.zeros((3, ny, nx))
o_mask = np.zeros_like(u[0],dtype=bool)
#Set obstacle
#o_mask[40:80, 40:80] = True

#Time loop
time_collector = 0
for t_idx in range(nt):
    plt.cla()
    plt.title(f't={t_idx*dt:.2f} s')
    #Inject source signal
    u[0, int(0.75*ny), int(0.15*nx)] = signal[t_idx]
    #Calculate next time step
    start_time = time.time()
    u = fd_step_varden(u, dt, dx, v, rho,
        method = 1, mur_bc = True, mur_rim_width=1)
    time_collector += time.time() - start_time
    #Apply obstacle on next time step
    u[0, o_mask] = 0
    uplot = np.squeeze(u[0].copy())
    uplot[o_mask] = np.nan
    extent = [-0.5*nx*dx, 0.5*nx*dx,
                -0.5*ny*dx, 0.5*ny*dx]
    plt.imshow(uplot, cmap=cmap, extent=extent)
    plt.clim(-plot_max, plot_max)
    plt.plot([-0.5*nx*dx, 0.5*nx*dx], [0, 0], color=3*[.75])
    ax.set_aspect('equal')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.pause(0.001)
    if t_idx == nt-1:
        plt.savefig('___figs/p250_2D_we_acoustic_vdensity.png', dpi=300, bbox_inches='tight')
        plt.pause(2)
print(f'Average time per time step: {1000*time_collector/nt:.3f} ms')
