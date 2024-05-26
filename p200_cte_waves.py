import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def fd_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    v: float,
    method: str = "numpy_indexing",
    mur_bc: bool = True,
    mur_rim_width: int = 3,
) -> np.ndarray:
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

    if method == "loop":
        u[2] = u[1]  # now -> old
        u[1] = u[0]  # next -> now
        for j in range(1, ny - 1):
            for k in range(1, nx - 1):
                u[0, j,
                  k] = alpha2 * (u[1, j + 1, k] + u[1, j - 1, k] +
                                 u[1, j, k + 1] + u[1, j, k - 1] - 4 *
                                 u[1, j, k]) + (2 * u[1, j, k] - u[2, j, k])

    elif method == "numpy_indexing":
        u[2] = u[1]  # now -> old
        u[1] = u[0]  # next -> now
        u[0, 1:ny - 1, 1:nx - 1] = (v * dt / dx)**2 * (
            u[1, 0:ny - 2, 1:nx - 1] + u[1, 2:ny, 1:nx - 1] +
            u[1, 1:ny - 1, 0:nx - 2] + u[1, 1:ny - 1, 2:nx] -
            4 * u[1, 1:ny - 1, 1:nx - 1]) + (2 * u[1, 1:ny - 1, 1:nx - 1] -
                                             u[2, 1:ny - 1, 1:nx - 1])
    elif method == "index_helpers":
        ip1, i, im1, jp1, j, jm1, kp1, k, km1 = index_helpers(u)
        u[im1] = u[i]  # now -> old
        u[i] = u[ip1]  # next -> now
        u[ip1, j[:, None],
          k] = (v * dt /
                dx)**2 * (u[i, jp1[:, None], k] + u[i, jm1[:, None], k] +
                          u[i, j[:, None], kp1] + u[i, j[:, None], km1] -
                          4 * u[i, j[:, None], k]) + (2 * u[i, j[:, None], k] -
                                                      u[im1, j[:, None], k])
    elif method == "convolution":
        u[2] = u[1]  # now -> old
        u[1] = u[0]  # next -> now
        stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        u[0] = signal.convolve2d(u[1], stencil, mode="same", boundary="wrap")
        u[0] = alpha2 * u[0] + 2 * u[1] - u[2]

    else:
        raise ValueError(
            f"Unknown method: {method}. Available methods are 'numpy_indexing' and 'index_helpers'."
        )

    if mur_bc:
        kappa = dt * v / dx
        top, bot = ny - 1, 0
        right, left = nx - 1, 0
        rim = mur_rim_width

        u[0, ny - rim - 1:top,
          1:nx - 1] = u[1, ny - rim - 2:top - 1, 1:nx - 1] + (kappa - 1) / (
              kappa + 1) * (u[0, ny - rim - 2:top - 1, 1:nx - 1] -
                            u[1, ny - rim - 1:top, 1:nx - 1])
        u[0, bot:rim, 1:nx -
          1] = u[1, bot + 1:rim + 1, 1:nx - 1] + (kappa - 1) / (kappa + 1) * (
              u[0, bot + 1:rim + 1, 1:nx - 1] - u[1, bot:rim, 1:nx - 1])
        u[0, 1:ny - 1, nx - 1 -
          rim:right] = u[1, 1:ny - 1, nx - 2 - rim:right - 1] + (kappa - 1) / (
              kappa + 1) * (u[0, 1:ny - 1, nx - 2 - rim:right - 1] -
                            u[1, 1:ny - 1, nx - 1 - rim:right])
        u[0, 1:ny - 1, left:rim] = u[1, 1:ny - 1, left + 1:rim + 1] + (
            kappa - 1) / (kappa + 1) * (u[0, 1:ny - 1, left + 1:rim + 1] -
                                        u[1, 1:ny - 1, left:rim])

    else:
        # Dirichlet boundary conditions
        u[ip1, 0, k] = 0
        u[ip1, ny - 1, k] = 0
        u[ip1, j, 0] = 0
        u[ip1, j, nx - 1] = 0
    return u


def fd_step_varden(
    u: np.ndarray,
    dt: float,
    dx: float,
    v: np.ndarray,
    rho: np.ndarray,
    method: str = 1,
    mur_bc: bool = True,
    mur_rim_width: int = 3,
) -> np.ndarray:
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

    u[2] = u[1]  # now -> old
    u[1] = u[0]  # next -> now

    if method == "loop":
        for (j, k), _ in np.ndenumerate(u[0]):
            if j == 0 or j == ny - 1 or k == 0 or k == nx - 1:
                pass
            else:
                u[0, j, k] = (2 * u[1, j, k] - u[2, j, k] +
                              (rho[j, k] * v[j, k]**2 * dt**2) / (2 * dx**2) *
                              ((1 / rho[j + 1, k] + 1 / rho[j, k]) *
                               (u[1, j + 1, k] - u[1, j, k]) -
                               (1 / rho[j, k] + 1 / rho[j - 1, k]) *
                               (u[1, j, k] - u[1, j - 1, k]) +
                               (1 / rho[j, k + 1] + 1 / rho[j, k]) *
                               (u[1, j, k + 1] - u[1, j, k]) -
                               (1 / rho[j, k] + 1 / rho[j, k - 1]) *
                               (u[1, j, k] - u[1, j, k - 1])))
    elif method == "index_helpers":
        # file:///home/bergmann/Downloads/applsci-13-08852.pdf
        # Simulation of Wave Propagation Using Finite Differences in Oil Exploration
        # variable density
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

        ip1, i, im1, jp1, j, jm1, kp1, k, km1 = index_helpers(u)
        u[ip1, j[:, None],
          k] = (2 * u[i, j[:, None], k] - u[im1, j[:, None], k] +
                (rho[j[:, None], k] * v[j[:, None], k]**2 * dt**2) /
                (2 * dx**2) *
                ((1 / rho[jp1[:, None], k] + 1 / rho[j[:, None], k]) *
                 (u[i, jp1[:, None], k] - u[i, j[:, None], k]) -
                 (1 / rho[j[:, None], k] + 1 / rho[jm1[:, None], k]) *
                 (u[i, j[:, None], k] - u[i, jm1[:, None], k]) +
                 (1 / rho[j[:, None], kp1] + 1 / rho[j[:, None], k]) *
                 (u[i, j[:, None], kp1] - u[i, j[:, None], k]) -
                 (1 / rho[j[:, None], k] + 1 / rho[j[:, None], km1]) *
                 (u[i, j[:, None], k] - u[i, j[:, None], km1])))

    if mur_bc:
        # Mur absorbing boundary conditions
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
        # Dirichlet boundary conditions
        u[ip1, 0, k] = 0
        u[ip1, ny - 1, k] = 0
        u[ip1, j, 0] = 0
        u[ip1, j, nx - 1] = 0
    return u


def index_helpers(u: np.ndarray) -> tuple:
    """Index helpers for the finite difference scheme."""
    # Time dimension
    ip1 = np.array([0])  # next, u[0]
    i = np.array([1])  # now, u[1]
    im1 = np.array([2])  # old, u[2]
    _, ny, nx = u.shape

    # Spatial dimension: y axis
    # j = 1:ny-1 (center)
    j = np.arange(ny)
    # j = 0:ny-2 (upper neighbour)
    jm1 = np.arange(0, ny - 1)
    jm1 = np.insert(jm1, 0, 0)
    # j = 2:ny (lower neighbour)
    jp1 = np.arange(1, ny)
    jp1 = np.insert(jp1, ny - 1, ny - 1)

    # Spatial dimension: x axis
    # k = 1:nx-1 (center)
    k = np.arange(nx)
    # k = 0:nx-2 (left neighbour)
    km1 = np.arange(0, nx - 1)
    km1 = np.insert(km1, 0, 0)
    # k = 2:nx (right neighbour)
    kp1 = np.arange(1, nx)
    kp1 = np.insert(kp1, nx - 1, nx - 1)
    return ip1, i, im1, jp1, j, jm1, kp1, k, km1


def source_signal(t_vec: np.ndarray,
                  A: float,
                  f: float,
                  source_type: str,
                  plot: bool,
                  delay: float = 0) -> np.ndarray:
    """Create a source signal for the 2D wave equation."""
    omega = 2 * np.pi * f

    #Apply delay
    t_vec = t_vec - delay
    if source_type == "ricker":
        # Ricker wavelet
        # shift t_vec to
        _t_vec = t_vec - 1.5 / f
        source_wavelet = (A * (1 - 2 * (np.pi * f * _t_vec)**2) *
                          np.exp(-((np.pi * f * _t_vec)**2)))
    elif source_type == "pulsed_sine":
        source_wavelet = (A * np.sin(t_vec * omega) *
                          np.clip(np.sin(0.5 * t_vec * omega), 0, 1))
    elif source_type == "continuous_sine":
        source_wavelet = A * np.sin(t_vec * omega)
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. Available source types are 'ricker', 'pulsed_sine' and 'continuous_sine'."
        )

    if plot:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(t_vec, source_wavelet)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Amplitude")
        plt.show()
    return source_wavelet


def source_bandlimited_random(
    t_vec: np.ndarray,
    A: float,
    f_min: float,
    f_max: float,
    plot: bool,
    alpha: float = 0.1,
) -> np.ndarray:
    """Create a bandlimited random source signal for the 2D wave equation."""
    nt = len(t_vec)

    spike_set = 2*(np.random.rand(nt) - .5)
    # bandpassfilter of the spike set
    spike_set = np.fft.fft(spike_set)
    freqs = np.fft.fftfreq(nt, d=t_vec[1] - t_vec[0])
    spike_set[(freqs < f_min)] = 0
    spike_set[(freqs > f_max)] = 0
    spike_set = np.fft.ifft(spike_set)
    source_wavelet = np.real(spike_set)
    source_wavelet = A * source_wavelet / np.max(np.abs(source_wavelet))
    window = signal.windows.tukey(len(t_vec), alpha=0.05)
    source_wavelet = source_wavelet * window
    if plot:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(t_vec, source_wavelet)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Amplitude")
        plt.show()

    return np.real(source_wavelet)


class Receiver:

    def __init__(self,
                 nr,
                 t_vec,
                 rec_x_idcs,
                 rec_y_idcs,
                 t0: float = 0,
                 rec_x0: float = 0,
                 rec_dx: float = 0):
        self.t_vec = t_vec  # time vector
        self.nr = nr  # number of receivers
        self.rec_x_idcs = rec_x_idcs  # x position of receivers, grid indices
        self.rec_y_idcs = rec_y_idcs  # y position of receivers, grid indices
        self.scan = np.zeros((len(t_vec), nr))
        self.t0 = t0
        self.rec_x0 = rec_x0
        self.rec_dx = rec_dx

    def record_step(self, t_idx, u):
        for rec_i in range(self.nr):
            # snap wavefield from nearest grid point and save to self.scan
            self.scan[t_idx, rec_i] = u[0, self.rec_y_idcs[rec_i],
                                        self.rec_x_idcs[rec_i]]

    def plot_rec_position(
        self,
        ax,
        model_x_vec: np.ndarray = None,
        model_y_vec: np.ndarray = None,
        color=[1, 0, 0],
    ):
        for rec_i in range(self.nr):
            ax.plot(
                model_x_vec[self.rec_x_idcs[rec_i]],
                model_y_vec[self.rec_y_idcs[rec_i]],
                "s",
                color=color,
            )

    def plot_scan(self, title: str = "", save_path: str = None, cmap="bwr", vmax=None):
        fig, ax = plt.subplots()
        if vmax is None:
            vmax = 0.95 * np.max(np.abs(self.scan))
        ax.imshow(
            self.scan,
            aspect="auto",
            extent=[
                0, self.nr - 1, self.t_vec[-1] - self.t0,
                self.t_vec[0] - self.t0
            ],
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            interpolation='nearest',
        )
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def get_grid_indices(self, u_x, u_y):
        x_idx = np.zeros(self.nr, dtype=int)
        y_idx = np.zeros(self.nr, dtype=int)
        for rec_idx in range(self.nr):
            # snap wavefield from nearest grid point and save to self.scan
            x_idx[rec_idx] = np.argmin(np.abs(self.rec_x_idcs[rec_idx] - u_x))
            y_idx[rec_idx] = np.argmin(np.abs(self.rec_y_idcs[rec_idx] - u_y))
        return x_idx, y_idx
