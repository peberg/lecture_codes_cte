import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey
from p200_cte_waves import fd_step, index_helpers, source_signal, source_bandlimited_random, Receiver


def dip_plane_model(nx: int, ny: int,
    left_idx: int = 10,
    right_idx: int = 30,
    plot = False):

    hor = tukey(int(0.5*nx), alpha=0.9)
    hor = np.append([np.zeros(int(0.25*nx))], hor)
    hor = np.append(hor, [np.zeros(nx-len(hor))])
    assert len(hor) == nx

    hor *= depth_in_indices
    hor += mean_depth_in_indices

    mod = np.zeros((ny, nx))
    for i in range(nx):
        mod[int(hor[i]):, i] = 1 

    if plot:
        plt.imshow(u.T)
        plt.colorbar()
        plt.show()
    return mod

def dented_model(nx: int, ny: int,
    depth_in_indices: int, 
    mean_depth_in_indices: int, plot = False):
    
    hor = tukey(int(0.5*nx), alpha=0.9)
    hor = np.append([np.zeros(int(0.25*nx))], hor)
    hor = np.append(hor, [np.zeros(nx-len(hor))])
    assert len(hor) == nx

    hor *= depth_in_indices
    hor += mean_depth_in_indices

    mod = np.zeros((ny, nx))
    for i in range(nx):
        mod[int(hor[i]):, i] = 1 

    if plot:
        plt.imshow(u.T)
        plt.colorbar()
        plt.show()
    return mod


#Set parameters
A = 80 #amplitude
f = 150 #Hz
v = 200 #m/s
omega = 2 * np.pi * f
dt = .0002 #s
nt = 400
dx = .1 #m
nx, ny = 220, 70
print('CFL: ', (v*dt)/dx)

#Create model space
model_x_vec = np.arange(-0.5*nx*dx, 0.5*nx*dx, dx)
model_y_vec = np.arange(0, ny*dx, dx)
assert len(model_x_vec) == nx
assert len(model_y_vec) == ny
t_vec = np.arange(0, dt*nt, dt)

#, 'dented'
for model_type in ['void', 'flat', 'hole']: 
    #Initialize wave field array and set obstacles
    u = np.zeros((3, ny, nx))
    #o_mask = np.zeros_like(u[0],dtype=bool)
    if model_type == 'void':
        o_mask = np.zeros_like(u[0]).astype(bool)
    elif model_type == 'flat':
        o_mask = dented_model(nx, ny, depth_in_indices=0, mean_depth_in_indices=35)
        o_mask = o_mask.astype(bool)
    elif model_type == 'hole':
        o_mask = np.zeros_like(u[0]).astype(bool)
        o_mask[int(0.6*ny)-3:int(0.6*ny)+3,
            int(0.5*nx)-3:int(0.5*nx)+3] = True
    elif model_type == 'dented':
        o_mask = dented_model(nx, ny, depth_in_indices=30, mean_depth_in_indices=35)
        o_mask = o_mask.astype(bool)

    #Create source signal
    signal = source_signal(t_vec, A, f, 
        source_type = 'ricker', plot = False)

    #Create receiver object
    nr = 40 #number of receivers
    rec = Receiver(nr, t_vec, 
            np.linspace(0.1*nx, 0.9*nx, nr, dtype=int), 
            np.full(nr, 15, dtype=int))

    #Figure for wavefield plotting
    fig, ax = plt.subplots()
    ax.axis("off")
    plot_max = 0.4*A
    cmap = plt.cm.bwr
    cmap.set_bad('gray')
    space_domain_extent = [-0.5*nx*dx, 0.5*nx*dx, ny*dx, 0]

    time_collector = 0 #for profiling
    for t_idx in range(nt):
        plt.cla()
        plt.title(f'{model_type} model, t={t_idx*dt:.2f} s')

        #Inject source signal
        #if signal is not too small inject it
        if np.max(np.abs(signal[t_idx])) > 0.001:
            for xi in range(int(0.1*nx), int(0.9*nx), 1):
                u[0, int(0.1*ny), xi] = signal[t_idx]
        
        #Calculate next time step
        start_time = time.time()
        u = fd_step(u, dt, dx, v, mur_bc = True)
        time_collector += time.time() - start_time
        rec.record_step(t_idx, u)

        #Apply obstacle on next time step
        u[0, o_mask] = 0
        uplot = np.squeeze(u[0].copy())
        uplot[o_mask] = np.nan

        #Plot wavefield
        plt.imshow(uplot, cmap=cmap, 
            extent=space_domain_extent)
        plt.clim(-plot_max, plot_max)
        ax.set_aspect('equal')
        ax = plt.gca()
        rec.plot_rec_position(ax, model_x_vec, model_y_vec, color = [0,0,0])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

        #save figure
        if t_idx == nt-1:
            plt.savefig(f'___figs/p560a_plane_wave_imaging_{model_type}.png', 
                dpi=300, bbox_inches='tight')
        plt.pause(0.001)

    rec.plot_scan(title = f'{model_type} model',
        save_path = f'___figs/p560b_plane_wave_imaging_{model_type}_data.png')
    print(f'Average time per time step: {1000*time_collector/nt:.3f} ms')

