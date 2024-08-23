# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:56:52 2024

HYPERION is an Open-Source code for calculating heat flux in axisymmetric
divertors.

It uses an implicit finite-difference method to estimate heat flux for given
thermal properties and geometric coordinates.

The file inputs are .csv files.

Depending on the need for the use of the heat transmission coefficient, alpha,
non-central difference or central difference methods can be used @ the surface.

The following is a script that outputs the heat flux as a funtion of time.

@author: P.O. Adebayo-Ige
"""

from temp_dep_prop import k_carbon, cp_temp
import numpy as np
import math as mt
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter

# Files from IR data
time_file = 'time132406.csv'
position_file = 'r132406.csv'
temp_file = 'temp132406.csv'

# Heat flux results for comparison with HYPERION
q_file = 'q_132406.csv'


# =============================================================================
# Building the mesh
# =============================================================================

times = []  # stores time values
# This gets the time increments from the csv file
with open(time_file) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        times.append(float(row[0]))
time_frames = len(times)  # Total Number of time frames
t = times[1] - times[0]

radius_m = []
# This gets the radius increments or change in r from the csv file
with open(position_file) as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        radius_m.append(float(row[0]))
r_nodes = len(radius_m)  # number of nodes in the radial (r) direction
r = radius_m[1] - radius_m[0]  # Mesh size units in r, or dr

# mesh size in y (dy), units are meters.
y_nodes = 22
y_length = 0.028
y = y_length / (y_nodes - 1)  # Mesh thickness dy or delta_y in meters

alpha = 6e4  # heat transmission coefficient (W/m^2-K)
w = 1  # Surface layer correction. Makes poorly adhered surface layer thin.

# =============================================================================
# Implicit Heat Flux Calculation & TACO Data
# =============================================================================

# The following are thermal constants and properties for ATJ Graphite
k = 93.27  # Heat Conductivity (W/m-K)
rho = 1760.1  # density (kg/m^3)
cp = 713.132  # Specific Heat Capacity (J/kg-K)


# ~~~~~~~~~~ Extracting Temperature and heat flux data ~~~~~~~~~~

# The entire 1D Temperature data from NSTX will be stored in this matrix.\
# Note that these are surface temperatures. No depth dimension, only radial.n
init_temp = np.zeros((r_nodes, time_frames))
temp_1d = np.zeros((r_nodes, time_frames))
taco_flux = np.zeros((r_nodes, time_frames))

with open(temp_file) as csv_file:
    csv_temp = csv.reader(csv_file)
    i = 0
    for row in csv_temp:
        span = len(row)  # the amount of nodes in total depth
        if i < time_frames:
            for j in range(span):
                # This remakes the csv data in python. Note it is "transposed"
                # by switching i & j. Look at how your data is formatted and
                # decide if you need to transpose.
                init_temp[j, i] = float(row[j])
        else:
            break
        i += 1

with open(q_file) as csv3_file:
    csv_heatflux = csv.reader(csv3_file)
    i = 0
    for row in csv_heatflux:
        width = len(row)
        for j in range(width):
            taco_flux[j, i] = float(row[j])
        i += 1


# ~~~~~~~~ 2D Calculations ~~~~~~~~~~~~~~~

# 2D temperature distribution vs time. shape(depth,surface pixels,timesteps)
# or shape(y,r,time) or shape(y_nodes,r_nodes,time). This is a 3D array.
temp_2d = np.zeros((y_nodes, r_nodes, time_frames))

# Initialize the 2D temperature array as the same initial temp as the 1D array.
# This can be changed to room temperature or another value.
temp_2d[:, :, 0] = init_temp[:, 0]

# Coefficient matrix. Recall Ax = b in  matrix algebra. This matrix is A.
m = np.zeros((r_nodes*y_nodes, r_nodes*y_nodes))

# The future temperatures that need to be calculated. Recall Ax = b, this is x.
temp_right = np.zeros(y_nodes*r_nodes)

# Temps from previous time steps. Recall Ax = b, this vector is b.
temp_left = np.zeros(y_nodes*r_nodes)


for j in range(1, time_frames):

    # First Layer with the heat transmission coefficient. This is the top of
    # the divertor surface. The temperatures of this top layer are detected by
    # the IR camera. They correspond to temperatures of the surface layer that
    # has properties described by alpha.
    for i in range(r_nodes):
        k_avg = (k_carbon(temp_2d[0, i, j-1]) + k_carbon(temp_2d[1, i, j-1]))/2
        m[i, i] = 1 + k_avg/(alpha*(y/w))
        m[i, i+r_nodes] = -k_avg/(alpha*(y/w))

    # Second Layer Non-Central difference methods
    # Understand that this is the 2nd layer into the depth of the "divertor",
    # but is technically the start of bulk slab. The y-index in temp_2d is 1.
    for i in range(r_nodes+1, 2*r_nodes-1):
        # toi = temperature of interest; used for temp-dependent properties
        toi = temp_2d[1, i-r_nodes, j-1]
        TDiff = k_carbon(toi)/(rho*cp_temp(toi))  # Thermal Diffusivity
        m[i, i-r_nodes] = -(2*w**2/(w+1))*t * TDiff/y**2    # North
        m[i, i+r_nodes] = -(2*w/(w+1))*t * TDiff/y**2       # South
        m[i, i] = 1 + t*TDiff*((2*w/y**2) + (2/r**2))       # Node of interest
        m[i, i+1] = -TDiff*(t/r**2)                         # East
        m[i, i-1] = -TDiff*(t/r**2)                         # West

    # Boundary points of first-second layer interface
    toi = temp_2d[1, 0, j-1]
    TDiff = k_carbon(toi)/(rho*cp_temp(toi))
    m[r_nodes, r_nodes] = 1 + t * TDiff*((2*w/y**2) + (2/r**2))
    m[r_nodes, 0] = -(2*w**2/(w+1))*t*TDiff/y**2
    m[r_nodes, 2*r_nodes] = -(2*w/(w+1))*t*TDiff/y**2
    m[r_nodes, r_nodes+1] = -2*TDiff*(t/r**2)

    toi = temp_2d[1, r_nodes-1, j-1]
    TDiff = k_carbon(toi)/(rho*cp_temp(toi))
    m[2*r_nodes-1, 2*r_nodes-1] = 1 + t * TDiff*((2*w/y**2) + (2/r**2))
    m[2*r_nodes-1, r_nodes-1] = -(2*w**2/(w+1))*t*TDiff/y**2
    m[2*r_nodes-1, 3*r_nodes-1] = -(2*w/(w+1))*t*TDiff/y**2
    m[2*r_nodes-1, 2*r_nodes-1 - 1] = -2*TDiff*(t/r**2)

    # Heat conduction in inner Graphite, regular central difference method
    # Understand that this is for layers between the 2nd layer and the bottom.
    for i in range(2, y_nodes-1):
        for u in range(1, r_nodes-1):
            toi = temp_2d[i, u, j-1]
            TDiff = k_carbon(toi)/(rho*cp_temp(toi))
            m[i*r_nodes + u, i*r_nodes + u - r_nodes] = -TDiff*(t/y**2)
            m[i*r_nodes + u, i*r_nodes + u + r_nodes] = -TDiff*(t/y**2)
            m[i*r_nodes + u, i*r_nodes + u] = 1 + TDiff*t*((2/y**2) + (2/r**2))
            m[i*r_nodes + u, i*r_nodes + u + 1] = -TDiff*(t/r**2)
            m[i*r_nodes + u, i*r_nodes + u - 1] = -TDiff*(t/r**2)

    # Boundary condition for heat insulation.
    for i in range(2, y_nodes-1):
        # left edge condition
        toi = temp_2d[i, 0, j-1]
        TDiff = k_carbon(toi)/(rho*cp_temp(toi))
        m[i*r_nodes, i*r_nodes - r_nodes] = -TDiff*(t/y**2)
        m[i*r_nodes, i*r_nodes + r_nodes] = -TDiff*(t/y**2)
        m[i*r_nodes, i*r_nodes] = 1 + TDiff*t*((2/y**2) + (2/r**2))
        m[i*r_nodes, i*r_nodes + 1] = -2*TDiff*(t/r**2)

        # Right Edge Condition
        toi = temp_2d[i, r_nodes-1, j-1]
        TDiff = k_carbon(toi)/(rho*cp_temp(toi))
        m[i*r_nodes + r_nodes - 1, i*r_nodes +
            r_nodes - 1 - r_nodes] = -TDiff*(t/y**2)
        m[i*r_nodes + r_nodes - 1, i*r_nodes +
            r_nodes - 1 + r_nodes] = -TDiff*(t/y**2)
        m[i*r_nodes + r_nodes - 1, i*r_nodes + r_nodes - 1] = 1 + \
            TDiff*t*((2/y**2) + (2/r**2))
        m[i*r_nodes + r_nodes - 1, i*r_nodes +
            r_nodes - 1 - 1] = -2*TDiff*(t/r**2)

    # Heat conduction at the bottom of divertor tile.
    for i in range(1, r_nodes-1):
        toi = temp_2d[y_nodes-1, i, j-1]
        TDiff = k_carbon(toi)/(rho*cp_temp(toi))
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i] = 1 + \
            TDiff*t*((2/y**2) + (2/r**2))
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1) *
          r_nodes + i - r_nodes] = -2*TDiff*(t/y**2)
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - 1] = -TDiff*(t/r**2)
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i + 1] = -TDiff*(t/r**2)

    # Here we are setting the boundary conditions not reached in previous loop.
    i = r_nodes - 1
    toi = temp_2d[y_nodes-1, i, j-1]
    TDiff = k_carbon(toi)/(rho*cp_temp(toi))
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i] = 1 + \
        TDiff*t*((2/y**2) + (2/r**2))
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1) *
      r_nodes + i - r_nodes] = -2*TDiff*(t/y**2)
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - 1] = -2*TDiff*(t/r**2)

    i = 0
    toi = temp_2d[y_nodes-1, i, j-1]
    TDiff = k_carbon(toi)/(rho*cp_temp(toi))
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i] = 1 + \
        TDiff*t*((2/y**2) + (2/r**2))
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1) *
      r_nodes + i - r_nodes] = -2*TDiff*t/y**2
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i + 1] = -2*TDiff*(t/r**2)
    
    # ~~~~~~~~~~ Matrix Algebra ~~~~~~~~~~~
        
    # Filling the 'b' vector in Ax = b. This pulls the IR camera temperatures.
    for i in range(r_nodes):
        temp_left[i] = init_temp[i, j]
    
    # Filling the 'b' vector in Ax = b. This pulls past temperatures simulated
    # by HYPERION. Temperatures from 2nd layer to last layer.
    for i in range(r_nodes):
        for k in range(1, y_nodes):
            temp_left[k*r_nodes + i] = temp_2d[k, i, j-1]
    
    # Solving linear equations
    temp_right = np.linalg.solve(m, temp_left)
    for k in range(y_nodes):
        for i in range(r_nodes):
            temp_2d[k, i, j] = temp_right[k*r_nodes + i]


# Calculate heat flux
heatflux = np.zeros((r_nodes, time_frames))
hflux_data = np.zeros((time_frames, 6))
for i in range(r_nodes):
    for j in range(time_frames):
        k_avg = (k_carbon(temp_2d[0, i, j]) + k_carbon(temp_2d[1, i, j]))/2
        heatflux[i, j] = (temp_2d[0, i, j] - temp_2d[1, i, j]) * k_avg/(y/w)

def integrate(y_vals, dx):  # y_vals is an array and dx is step_size
    if (len(y_vals) - 1) % 2 == 0:  # Simpson's 1/3 Rule [Preferred] – even number of intervals (odd # of points)
        i = 1
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            if i % 2 == 0:
                total += 2 * y
            else:
                total += 4 * y
            i += 1
        return total * (dx / 3.0)
    elif (len(y_vals) - 1) % 3 == 0:  # Simpson's 3/8 Rule – number of intervals is divisible by 3
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            total += 3 * y
        return total * (3 * dx / 8)
    else:  # Combo of 1/3 & 3/8 Rules for odd number of intervals
        total1 = (3 * dx / 8) * (y_vals[0] + 3*y_vals[1] + 3*y_vals[2] + y_vals[3])
        total2 = y_vals[4] + y_vals[-1]  # Simpson's 1/3 Rule
        i = 1
        for y in y_vals[5:-1]:
            if i % 2 == 0:
                total2 += 2 * y
            else:
                total2 += 4 * y
            i += 1
        total2 = total2 * (dx / 3.0)
        total = total1 + total2
        return total



# %% General Figures

fig_tit = 'NSTX #132406'  # Title of Primary Data Figures
fg2_tit = "TACO NSTX #132406"  # Title of Secondary Data Figures
dat_tit = 'HYPERION {} nodes, w={}'.format(y_nodes, w)
dt2_tit = 'TACO'
shot = '132406'

t_plt = np.array(times[0:time_frames])
t_dim = 2785
t1 = 0.4515  # Settting time domain of plots of interest. Uncommnent below to
t2 = 0.4535
r_plt = np.array(radius_m[0:r_nodes])
r_dim = 16  # the radial node of interest for heat flux evolution
taco_real_flux = taco_flux * 1e6  # in Watts not MW

# Heat Flux Color Map of HYPERION Calculated Values
plt.figure(1)
c = plt.pcolormesh(t_plt, r_plt, heatflux/1e6,
                   cmap='turbo', shading='gouraud')
cbar = plt.colorbar(c)
cbar.set_label(r'Heat Flux $\mathbf{(MW/m^2)}$', fontsize=14,
               weight='bold', rotation=270, labelpad=20, y=0.5)
cbar.ax.tick_params(labelsize=14)
# plt.xlim(t1, t2)  # Uncomment to set time domain
plt.clim(-1, 27)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('R (m)', fontsize=14, weight='bold')
plt.xlabel('Time (s)', fontsize=14, weight='bold')
plt.title('HYPERION {}'.format(fig_tit), weight='bold')

# Heat Flux Color Map of TACO or other sample code for comparison
plt.figure(2)
c = plt.pcolormesh(t_plt, r_plt, taco_real_flux/1e6,
                   cmap='turbo', shading='gouraud')
cbar = plt.colorbar(c)
cbar.set_label(r'Heat Flux $\mathbf{(MW/m^2)}$', fontsize=14,
               weight='bold', rotation=270, labelpad=20, y=0.5)
cbar.ax.tick_params(labelsize=14)
# plt.xlim(t1, t2)  # Uncomment to set time domain
plt.clim(-1, 27)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('R (m)', fontsize=14, weight='bold')
plt.xlabel('Time (s)', fontsize=14, weight='bold')
plt.title('{}'.format(fg2_tit), weight='bold')

# Temporal Heat Flux Evolution at a given pixel 'r_dim'
plt.figure(3)
plt.plot(t_plt, heatflux[r_dim]/1e6, 'k--', label=dat_tit)
plt.plot(t_plt, taco_flux, 'r:', label=dt2_tit)
plt.title('{} R ={:.4f}m'.format(fig_tit, radius_m[r_dim]), weight='bold')
plt.xlabel('Time (s)', fontsize=14, weight='bold')
plt.ylabel(r'Heat Flux $\mathbf{(MW/m^2)}$', fontsize=14, weight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left')
