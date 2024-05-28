import time
import numpy as np
import matplotlib.pyplot as plt
from p010_cte_waves import fd_step, index_helpers, source_signal, source_bandlimited_random

class Receiver:
    def __init__(self, nr, t_vec, rec_x_idcs, rec_y_idcs, t0: float = 0):
        self.t_vec = t_vec #time vector
        self.nr = nr #number of receivers
        self.rec_x_idcs = rec_x_idcs #x position of receivers, grid indices
        self.rec_y_idcs = rec_y_idcs #y position of receivers, grid indices
        self.scan = np.zeros((len(t_vec), nr))
        self.t0 = t0

    def record_step(self, t_idx, u):
        for rec_i in range(self.nr):
            #snap wavefield from nearest grid point and save to self.scan
            self.scan[t_idx, rec_i] = u[0,
                                        self.rec_y_idcs[rec_i],
                                        self.rec_x_idcs[rec_i]]

    def plot_rec_position(self, ax,
        model_x_vec: np.ndarray = None,
        model_y_vec: np.ndarray = None,
        color = [1, 0, 0]):
        for rec_i in range(self.nr):
            ax.plot(model_x_vec[self.rec_x_idcs[rec_i]],
                    model_y_vec[self.rec_y_idcs[rec_i]],
                    's', color = color)

    def plot_scan(self):
        fig, ax = plt.subplots()
        vmax = 0.95*np.max(np.abs(self.scan))
        ax.imshow(self.scan, aspect='auto',
            extent=[0, self.nr-1, self.t_vec[-1], self.t0],
            vmin=-vmax, vmax=vmax, cmap='bwr',
            interpolation=None)
        ax.set_xlabel('Receiver')
        ax.set_ylabel('Time (s)')
        plt.show()

    def get_grid_indices(self, u_x, u_y):
        x_idx = np.zeros(self.nr, dtype=int)
        y_idx = np.zeros(self.nr, dtype=int)
        for rec_idx in range(self.nr):
            #snap wavefield from nearest grid point and save to self.scan
            x_idx[rec_idx] = np.argmin(np.abs(self.rec_x_idcs[rec_idx] - u_x))
            y_idx[rec_idx] = np.argmin(np.abs(self.rec_y_idcs[rec_idx] - u_y))
        return x_idx, y_idx


A = 80 #amplitude
v = 200 #m/s
f = 150 #Hz
omega = 2 * np.pi * f

dt = .0002 #s
nt = 500 ##400
dx = .1 #m
nx, ny = 200, 100
model_x_vec = np.arange(-0.5*nx*dx, 0.5*nx*dx, dx)
model_y_vec = np.arange(0, ny*dx, dx)
assert len(model_x_vec) == nx
assert len(model_y_vec) == ny

source_type = 'ricker_single_source' 
#'uncorrelated_multi_source'
#'ricker_single_source'
#

#Create time vector
t_vec = np.arange(0, dt*nt, dt)
#'numpy_indexing' or 'index_helpers'
print('CFL: ', (v*dt)/dx)

nr = 40
rec = Receiver(nr, t_vec,
        np.linspace(0.1*nx, 0.9*nx, nr, dtype=int),
        np.full(nr, 3, dtype=int))

#Create source signal
if source_type == 'ricker_single_source':
    _signal = source_signal(t_vec, A, f, source_type='ricker', plot=False)
elif source_type == 'uncorrelated_multi_source':
    np.random.seed(10)
    signal1 = source_bandlimited_random(t_vec, A, f_min=10, f_max=200, plot=False)
    signal2 = source_bandlimited_random(t_vec, A, f_min=10, f_max=200, plot=False)
    signal3 = source_bandlimited_random(t_vec, A, f_min=10, f_max=200, plot=False)
else:
    raise ValueError(
        f"Unknown source type: {source_type}. Available source types are 'ricker_single_source' and 'uncorrelated_multi_source'."
    )

fig, ax = plt.subplots()
ax.axis("off")
plot_max = 0.15*A
cmap = plt.cm.bwr
cmap.set_bad('gray')
space_domain_extent = [-0.5*nx*dx, 0.5*nx*dx,
            ny*dx, 0]


u = np.zeros((3, ny, nx))
o_mask = np.zeros_like(u[0],dtype=bool)
time_collector = 0

for t_idx in range(nt):
    plt.cla()
    plt.title(f't={t_idx*dt:.2f} s')

    #Inject source signal
    if source_type == 'ricker_single_source':
        sources = [(0.65 * nx, 0.4 * ny)]
        signal = _signal
        for sou_idx, source in enumerate(sources):
            if not np.isclose(signal[t_idx], 0, atol=1e-8):
                u[0, int(source[1]), int(source[0])] = signal[t_idx]
    elif source_type == 'uncorrelated_multi_source':
        sources = [(0.3 * nx, 0.3 * ny), (0.5 * nx, 0.55 * ny),
                   (0.65 * nx, 0.4 * ny)]
        for sou_idx, source in enumerate(sources):
            signal = [signal1, signal2, signal3][sou_idx]
            if not np.isclose(signal[t_idx], 0, atol=1e-8):
                u[0, int(source[1]), int(source[0])] = signal[t_idx]

    #Calculate next time step
    start_time = time.time()
    u = fd_step(u, dt, dx, v, mur_bc=True)
    time_collector += time.time() - start_time
    rec.record_step(t_idx, u)

    #Apply obstacle on next time step
    o_mask = o_mask
    u[0, o_mask] = 0
    uplot = np.squeeze(u[0].copy())
    uplot[o_mask] = np.nan
    plt.imshow(uplot, cmap=cmap, extent=space_domain_extent)
    plt.clim(-plot_max, plot_max)
    ax.set_aspect('equal')
    ax = plt.gca()
    rec.plot_rec_position(ax, model_x_vec, model_y_vec, color=[0, 0, 0])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    #save figure
    if t_idx == nt - 1:
        plt.savefig(f'___figs/p550a_reverse_time_imaging.png',
                    dpi=300,
                    bbox_inches='tight')
    plt.pause(0.001)

rec.plot_scan()
print(f'Average time per time step: {1000*time_collector/nt:.3f} ms')



#Reverse-time imaging
scan = rec.scan
scan = np.flip(scan, axis=0)

fig, ax = plt.subplots()
ax.axis("off")
plot_max = 0.15*A
cmap = plt.cm.bwr
cmap.set_bad('gray')

u = np.zeros((3, ny, nx))
o_mask = np.zeros_like(u[0],dtype=bool)
time_collector = 0
amp_collector = np.zeros((ny, nx))

for t_idx in range(nt):
    plt.cla()
    plt.title(f't={t_idx*dt:.2f} s')

    #Inject source signal
    for i in range(nr):
        u[0, rec.rec_y_idcs[i], rec.rec_x_idcs[i]] = scan[t_idx, i]

    #Calculate next time step
    start_time = time.time()
    u = fd_step(u, dt, dx, v, mur_bc = True)
    amp_collector += u[0]**2
    time_collector += time.time() - start_time

    #Apply obstacle on next time step
    o_mask = o_mask
    u[0, o_mask] = 0
    uplot = np.squeeze(u[0].copy())
    uplot[o_mask] = np.nan
    plt.imshow(uplot, cmap=cmap, extent=space_domain_extent)
    plt.clim(-plot_max, plot_max)
    ax.set_aspect('equal')
    ax = plt.gca()
    rec.plot_rec_position(ax, model_x_vec, model_y_vec, color = [0,0,0])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    #save figure
    if t_idx == nt-1:
        plt.savefig(f'___figs/p550b_reverse_time_imaging.png',
            dpi=300, bbox_inches='tight')
    plt.pause(0.001)

fig, ax = plt.subplots()
ax.imshow(amp_collector, cmap='hot', extent=space_domain_extent, vmin = 0.3*np.max(amp_collector), vmax = 0.9*np.max(amp_collector))
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
rec.plot_rec_position(ax, model_x_vec, model_y_vec, color = [1,1,1])
for source in sources:
    ax.plot(model_x_vec[int(source[0])], model_y_vec[int(source[1])], 'b+', markersize=15)
plt.savefig(f'___figs/p550c_reverse_time_imaging.png',
    dpi=300, bbox_inches='tight')
plt.show()
