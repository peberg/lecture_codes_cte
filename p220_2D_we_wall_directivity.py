import numpy as np
import matplotlib.pyplot as plt
import time
from p200_cte_waves import fd_step, index_helpers, source_signal

scenario = 'A'
# 'A' = Single source
# 'B' = Single source with wall reflection
# 'C' = Single source with image source

A = 80 #amplitude
f = 120 #Hz
omega = 2 * np.pi * f
v = 200 #m/s

dt = .0002 #s
nt = 500
dx = .1 #m
nx, ny = 240, 240

#Create time vector
t_vec = np.arange(0, dt*nt, dt)
#'numpy_indexing' or 'index_helpers'
print('CFL: ',(v*dt)/dx)

#Create source signal
signal = source_signal(t_vec, A, f, 
    source_type = 'continuous_sine', plot = False)

fig, ax = plt.subplots()
ax.axis("off")
plot_max = 0.3*A
cmap = plt.cm.bwr
cmap.set_bad('gray')

u = np.zeros((3, ny, nx))
o_mask = np.zeros_like(u[0],dtype=bool)
if scenario == 'B':
    o_mask[65:-65, nx//2-4:nx//2+1] = True
time_collector = 0

for t_idx in range(nt):
    plt.cla()
    plt.title(f't={t_idx*dt:.2f} s')
    #Inject source signal

    if scenario in ['A', 'B']:
        u[0, ny//2, nx//2+28] =  signal[t_idx]
    elif scenario == 'C':
        u[0, ny//2, nx//2+28] =  signal[t_idx]
        u[0, ny//2, nx//2-28] = -signal[t_idx]
    
    #Calculate next time step
    start_time = time.time()
    u = fd_step(u, dt, dx, v,
        mur_bc = True)
    time_collector += time.time() - start_time
    #Apply obstacle on next time step
    o_mask = o_mask
    u[0, o_mask] = 0
    uplot = np.squeeze(u[0].copy())
    uplot[o_mask] = np.nan
    extent = [-0.5*nx*dx, 0.5*nx*dx,  
              -0.5*ny*dx, 0.5*ny*dx]
    plt.imshow(uplot, cmap=cmap, extent=extent)
    plt.clim(-plot_max, plot_max)
    ax.set_aspect('equal')
    ax = plt.gca()
    #ax.invert_yaxis()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    #save figure
    if t_idx == nt-1:
        plt.savefig(f'___figs/p300_wall_reflection_{scenario}.png', 
            dpi=300, bbox_inches='tight') ##
    plt.pause(0.001)

print(f'Average time per time step: {1000*time_collector/nt:.3f} ms')