import numpy as np
import matplotlib.pyplot as plt
from p010_cte_waves import fd_step, source_signal

A = 80 #amplitude
dt = .02 #s
dx = .1 #m
f = 2 #Hz
v=2 #m/s
omega = 2 * np.pi * f #angular frequency
nt = 800 #number of time steps
nx = 300 #number of grid points
ny = 200 #number of grid points
t_vec = np.arange(0, dt*nt, dt)
x_vec = np.arange(0, nx*dx, dx)
extent = [0, nx*dx, 0, ny*dx]

print('CFL:', (v*dt)/dx)
u = np.zeros((3, ny, nx))

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax2.axis('off')

plot_max = 1.4*A
amp_collector = np.zeros_like(u[0])

#Create source signals
source_idcs_x = np.arange((nx // 2) - 30, (nx // 2) + 30).astype(int)
source_idcs_y = 5 + np.zeros_like(source_idcs_x).astype(int)
source_signals = []
for i in range(len(source_idcs_x)):
    #define delay
    delay = -i * dt
    source_signals.append(
        source_signal(t_vec,
                      A,
                      f,
                      source_type='continuous_sine',
                      plot=False,
                      delay=delay))

for t_idx in range(nt):
    ax1.clear()
    #inject source signal
    for i in range(len(source_idcs_x)):
        u[0, source_idcs_y[i], source_idcs_x[i]] = source_signals[i][t_idx]
    u = fd_step(u, dt, dx, v, mur_bc=True)
    ax1.set_title(f't={t_idx*dt:.2f} s')
    ax1.imshow(u[0],
               cmap=plt.cm.bwr,
               vmin=-plot_max,
               vmax=plot_max,
               extent=extent)
    #Start collecting amplitudes after 10 seconds
    if t_idx * dt > 10.:
        amp_collector += u[0]**2
        ax2.axis('on')
        ax2.clear()
        ax2.imshow(amp_collector, cmap='afmhot', extent=extent)
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    #save figure
    if t_idx == nt-1:
        plt.savefig(f'___figs/p570_delay_laws2.png', 
            dpi=300, bbox_inches='tight') ##
    plt.pause(0.001)
