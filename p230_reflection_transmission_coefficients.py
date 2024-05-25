import numpy as np
import matplotlib.pyplot as plt

# Constants
rho1 = 1000  # Density of medium 1, in kg/m^3
rho2 = 1500  # Density of medium 2, in kg/m^3
V1 = 1500    # Velocity in medium 1, in m/s
V2 = 2500    # Velocity in medium 2, in m/s

# Acoustic impedances
Z1 = rho1 * V1
Z2 = rho2 * V2

# Angle range
theta_i_deg = np.linspace(0, 90, 500)  # Incident angles in degrees
theta_i = np.deg2rad(theta_i_deg)      # Convert degrees to radians

# Snell's law
theta_t = np.arcsin(V1 / V2 * np.sin(theta_i))  # Transmission angle

# Critical angle
theta_c = np.arcsin(V2 / V1)  # Critical angle
print('critical angle:', np.rad2deg(theta_c), ' degrees')

# Reflection and transmission coefficients (pressure)
R = (Z2 * np.cos(theta_i) - Z1 * np.cos(theta_t)) / \
    (Z2 * np.cos(theta_i) + Z1 * np.cos(theta_t))
T = 2 * Z2 * np.cos(theta_i) / \
    (Z2 * np.cos(theta_i) + Z1 * np.cos(theta_t))

# Plot
plt.figure()
plt.plot(theta_i_deg, R, label='R')
plt.plot(theta_i_deg, T, label='T')
plt.xlabel('Incident angle (degrees)')
plt.ylabel('Coefficient')
plt.xlim([0, 90])
plt.ylim([-0.5, 2])
plt.legend()

plt.text(60, 0, f'V1={V1} m/s\nV2={V2} m/s\nrho1={rho1} kg/m³\nrho2={rho2} kg/m³')

plt.show()
