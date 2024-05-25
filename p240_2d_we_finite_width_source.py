import numpy as np
import matplotlib.pyplot as plt
from p200_cte_waves import source_signal

def u_step(u, dt, dx, v, mur_bc = True):
    _, ny, nx = u.shape
    u[2] = u[1]     # old k -> new k-1
    u[1] = u[0]     # old k+1 -> new k

    alpha = v * dt / dx
    alpha2 = alpha**2

    # Calculate the new k+1:
    u[0, 1:ny-1, 1:nx-1]  = alpha2 * (
                u[1, 0:ny-2, 1:nx-1]
                + u[1, 2:ny,   1:nx-1]
                + u[1, 1:ny-1, 0:nx-2]
                + u[1, 1:ny-1, 2:nx]
                - 4*u[1, 1:ny-1, 1:nx-1]) \
                + (2 * u[1, 1:ny-1, 1:nx-1]
                - u[2, 1:ny-1, 1:nx-1])

    if mur_bc:
        # Mur absorbing boundary conditions.
        kappa = (1 - alpha) / (1 + alpha)
        u[0, 0, 1:nx-1] = (u[1, 1, 1:nx-1]
                            - kappa * (
                                    u[0, 1, 1:nx-1]
                                - u[1, 0, 1:nx-1])
                            )
        u[0, ny-1, 1:nx-1] = (u[1, ny-2, 1:nx-1]
                            + kappa * (
                                u[1, ny-1, 1:nx-1]
                                - u[0, ny-2, 1:nx-1])
                            )
        u[0, 1:ny-1, 0] = (u[1, 1:ny-1, 1]
                            - kappa * (
                                u[0, 1:ny-1, 1]
                            - u[1, 1:ny-1, 0])
                            )
        u[0, 1:ny-1, nx-1] = (u[1, 1:ny-1, nx-2]
                            + kappa * (
                                u[1, 1:ny-1, nx-1]
                                - u[0, 1:ny-1, nx-2])
                            )
    else:
        u[0, 0, 1:nx-1] = 0
        u[0, ny-1, 1:nx-1] = 0
        u[0, 1:ny-1, 0] = 0
        u[0, 1:ny-1, nx-1] = 0
    return u


source_type = 'continuous_wavelet' #'single_wavelet'
A = 80 #amplitude
dt = .02 #s
dx = .1 #m
f = 2 #Hz
v=2 #m/s
omega = 2 * np.pi * f #angular frequency

if source_type == 'continuous_wavelet':
    nt = 800 #number of time steps
    t_vec = np.arange(0, dt*nt, dt)
    signal = source_signal(t_vec,
                           A,
                           omega / (2 * np.pi),
                           source_type='continuous_sine',
                           plot=False)
    plot_max = 1.4 * A
elif source_type == 'single_wavelet':
    nt = 180 #number of time steps
    t_vec = np.arange(0, dt*nt, dt)
    signal = source_signal(t_vec,
                           A,
                           omega / (2 * np.pi),
                           source_type='ricker',
                           plot=False)
    plot_max = .03 * A


nx = ny = 200 #number of grid points
x_vec = np.arange(0, nx*dx, dx)
extent = [0, nx * dx, 0, ny * dx]
u = np.zeros((3, ny, nx))
print('CFL:', (v*dt)/dx)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#Hide ax2
ax2.axis('off')

amp_collector = np.zeros_like(u[0])

for i in range(nt):
    ax1.clear()
    #inject source signal
    u[0, 0, (nx // 2) - 30:(nx // 2) + 30] = signal[i]
    u = u_step(u, dt, dx, v, mur_bc = True)
    ax1.set_title(f't={i*dt:.2f} s')
    ax1.imshow(u[0],
               cmap=plt.cm.bwr,
               vmin=-plot_max,
               vmax=plot_max,
               extent=extent)
    #Start collecting amplitudes after 10 seconds
    if i * dt > 10.:
        amp_collector += u[0]**2
        ax2.axis('on')
        ax2.clear()
        ax2.imshow(amp_collector, cmap='afmhot', extent=extent)
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    plt.pause(0.001)
