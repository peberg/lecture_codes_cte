import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from p200_cte_waves import source_signal, fd_step, index_helpers, Receiver

#Model parameters
A = 80 #amplitude
f = 3e6 #150 #Hz
omega = 2 * np.pi * f
v = 5600 #30 #m/s
dt = 1e-8 #.0002 #s
nt = 600
nr = 40 #Number of receivers/sources
model_type = 'complex'
plot = True

#Model domain
x0, x1, dx = -.015, .015, 0.0001 #-1.5, 1.5, 0.01
y0, y1, dy = 0, .012, dx #0, 1, 0.01
x_vec = np.arange(x0, x1, dx)
y_vec = np.arange(y0, y1, dy)
y_vec = np.linspace(y1, y0, len(y_vec))
nx, ny = len(x_vec), len(y_vec)

if model_type == 'complex':
    #Set up wave field and obstacles
    o_mask = np.zeros((ny, nx), dtype=bool)
    o_mask[:70, :50+30] = True
    o_mask[20:70, 50+30:100+30] = np.rot90(np.triu(np.ones((50,50)))).astype(bool)
    o_mask[:45, :125+30] = True
    o_mask[20:45, 100+30:125+30] = np.rot90(np.triu(np.ones((25,25)))).astype(bool)
    o_mask[:20, 125+30:] = True

    o_mask[20:70, -100:-50] = np.triu(np.ones((50,50))).astype(bool)
    o_mask[:70, -50:] = True
    o_mask[:15, 50:-50] = True

    #Add circle
    circle_center_x = 190
    circle_center_y = 99
    circle_radius = 80
    _y, _x = np.ogrid[-circle_center_y:ny - circle_center_y,
                    -circle_center_x:nx - circle_center_x]
    mask = _x**2 + _y**2 <= circle_radius**2
    o_mask[mask] = False

    #Add circle
    circle_center_x = 190
    circle_center_y = 20
    circle_radius = 7
    _y, _x = np.ogrid[-circle_center_y:ny - circle_center_y,
                        -circle_center_x:nx - circle_center_x]
    mask = _x**2 + _y**2 <= circle_radius**2
    o_mask[mask] = True
elif model_type == 'diffractor':
    o_mask = np.zeros((ny,nx)).astype(bool)
    o_mask[int(0.6*ny)-3:int(0.6*ny)+3,
        int(0.5*nx)-3:int(0.5*nx)+3] = True

#Time vector
t_vec = np.arange(0, dt*nt, dt)
print('CFL: ',(v*dt)/dx)

rec_collector = []


for sou_idx in range(nr):
    print(f'Source {sou_idx+1}/{nr}')

    #Source signal
    signal = source_signal(t_vec, A, f, source_type='ricker', plot=False)

    #Create receiver object
    rec_x_idcs = np.linspace(0.1 * nx, 0.9 * nx, nr, dtype=int)
    rec_y_idcs = np.full(nr, int(0.85 * ny), dtype=int)
    rec_x = x_vec[rec_x_idcs]
    rec_y = y_vec[rec_y_idcs]
    rec = Receiver(nr,
                   t_vec,
                   rec_x_idcs,
                   rec_y_idcs,
                   rec_x0=rec_x[0],
                   rec_dx=rec_x[1] - rec_x[0])
    rec.t0 = dt * np.argmax(signal)

    u = np.zeros((3, ny, nx))
    #Source position
    sou_x_idx = rec.rec_x_idcs[sou_idx]
    sou_y_idx = rec.rec_y_idcs[sou_idx]

    if plot:
        fig, ax = plt.subplots()
        ax.axis("off")
        plot_max = 0.15 * A
        cmap = plt.cm.bwr
        cmap.set_bad('gray')

    #np.triu(np.ones_like(x_mat)).astype(bool)
    time_collector = 0

    for t_idx in range(nt):
        #Inject source signal
        if not np.isclose(signal[t_idx], 0, atol=1e-8):
            u[0, sou_y_idx, sou_x_idx] = signal[t_idx]
        #Calculate next time step
        start_time = time.time()
        u = fd_step(u, dt, dx, v, method='numpy_indexing', mur_bc=True)
        rec.record_step(t_idx, u)
        time_collector += time.time() - start_time
        #Apply obstacle on next time step
        u[0, o_mask] = 0
        uplot = np.squeeze(u[0].copy())
        uplot[o_mask] = np.nan

        if plot:
            plt.cla()
            plt.title(f't={1e6*t_idx*dt:.2f} $\mu$s')
            plt.imshow(uplot, cmap=cmap, extent=[x0, x1, y0, y1])
            plt.clim(-plot_max, plot_max)
            ax.set_aspect('equal')
            ax = plt.gca()
            rec.plot_rec_position(ax, x_vec, y_vec, color=[0, 0, 0])
            ax.invert_yaxis()
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            plt.pause(0.001)
    if plot:
        plt.close()

    print(f'   Average time per time step: {1000*time_collector/nt:.3f} ms')
    rec_collector.append(rec)
    if plot:
        rec.plot_scan(cmap=cmap)

#Save rec_collector
with open('___datasets/p510_FMC_rec_dataset.pkl', 'wb') as f:
    pickle.dump(rec_collector, f)

model_parms = dict(A=A,
                   o_mask = o_mask,
                   x0=x0, x1=x1, dx=dx,
                   y0=y0, y1=y1, dy=dy,
                   x_vec=x_vec,
                   y_vec=y_vec,
                   v=v,
                   dt=dt,
                   nt=nt,
                   nx=nx, ny=ny,
                   rec_x=rec_x, rec_y=rec_y,
                   omega=omega)
#Save model_parms to pickle
with open('___datasets/p510_FMC_model_parms.pkl', 'wb') as f:
    pickle.dump(model_parms, f)
