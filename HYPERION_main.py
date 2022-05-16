# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:01:10 2022

HYPERION is an Open-Source code for calculating heat flux in axisymmetric
divertors.

It uses an implicit finite-difference method to estimate heat flux for given
thermal properties and geometric coordinates.

The file inputs can be csv data or you can setup your own geometry.

Depending on the need for the use of the heat transmission coefficient, alpha,
non-central difference or central difference methods can be used @ the surface.


@author: P.O. Adebayo-Ige
"""

from temp_dep_prop import k_carbon, cp_temp
import numpy as np
import math as mt
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import savemat

# =============================================================================
# Building the mesh
# =============================================================================
time_file = 'time132406.csv'
position_file = 'r132406.csv'
temp_file = 'temp132406.csv'
flux_expan = 'fx132406.csv'


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
y_nodes = int(
    input('What is the number of nodes in your y (vertical) dimension?'))
y_length = float(input(
    'What is the thickness of the slab in y dimension (meters)?'))

# Mesh thickness dy or delta_y in meters in the bulk. Recall thin mesh at top.
y = y_length / (y_nodes - 1)
w = 10
y_srf = y/w  # Mesh size at surface layer
alpha = 6e4  # heat transmission coefficient (MW/m^2-K)


# =============================================================================
# Implicit Heat Flux Calculation
# =============================================================================

# The following are thermal constants and properties of graphite. Change if
# req'd, or create a function like k_carbon and cp_temp for your slab material.
k = 93.27  # Heat Conductivity (W/m-K)
rho = 1760.1  # density (kg/m^3)
cp = 713.132  # Specific Heat Capacity (J/kg-K)


# ~~~~~~~~~~ Extracting Temperature and heat flux data ~~~~~~~~~~

# The entire 1D Temperature data from NSTX will be stored in this matrix.
# Note that these are SURFACE temperatures. No depth dimension, only radial. So
# each temperature in a column is a different radial position on the surface.
init_temp = np.zeros((r_nodes, time_frames))
temp_1d = np.zeros((r_nodes, time_frames))

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


# The following loop looks at the temperatures and subtracts the initial (t=0)
# temperature difference between the near & far edges of the slab surface from
# the position of interest. This rids of negative heat flux that is observed in
# steady-state. Note that all temperatures at t=0 should be almost the same.
# Depending on the accuracy of your data, this may or may not be needed.

for i in range(r_nodes):
    for j in range(time_frames):
        temp_1d[i, j] = init_temp[i, j] - \
            (init_temp[i, 0] - init_temp[r_nodes-1, 0])

# ~~~~~~~~ 2-D Calculations ~~~~~~~~~~~~~~~

# 2D temperature distribution vs time. shape(depth,rows/surface pixels,columns)
# or shape(y,r,time) or shape(y_nodes,r_nodes,time). This is a 3D array.
temp_2d = np.zeros((y_nodes, r_nodes, time_frames))

# Make the 2D the same initial temperature as the 1D array. Note that you're
# taking the surface temperature from the second column in the temp_1d matrix
temp_2d[:, :, 0] = temp_1d[1, 0]
m = np.zeros((r_nodes*y_nodes, r_nodes*y_nodes))  # Coefficient matrix.

# Temps from previous time steps. Recall Ax = b in matrix algebra, this is b.
temp_left = np.zeros(y_nodes*r_nodes)

# The future temperatures that need to be calculated. Recall Ax = b, this is x.
temp_right = np.zeros(y_nodes*r_nodes)

# The temperatures on all nodes. This is just to store ALL the temperatures.
# It isn't used in any calcs... use it as you see fit.
templeft_all = np.zeros((y_nodes*r_nodes, time_frames))

for j in range(1, time_frames):

    # First Layer with the heat transmission coefficient
    for i in range(r_nodes):
        m[i, i] = 1 + k_carbon(temp_2d[0, i, j-1])/(alpha*(y/w))
        m[i, i+r_nodes] = -k_carbon(temp_2d[0, i, j-1])/(alpha*(y/w))

    # Second Layer Non-Central difference methods
    # Understand that this is the 2nd layer of the slab. The y-index in
    # temp_2d is 1.
    for i in range(r_nodes+1, 2*r_nodes-1):
        # toi = temperature of interest; used for temp-dependent properties
        toi = temp_2d[1, i-r_nodes, j-1]
        m[i, i-r_nodes] = -(2*w**2/(w+1))*t * \
            (k_carbon(toi)/(rho*cp_temp(toi)))/y**2
        m[i, i+r_nodes] = -(2*w/(w+1))*t*(k_carbon(toi
                                                   )/(rho*cp_temp(toi)))/y**2
        m[i, i] = 1 + t*(k_carbon(toi)/(rho *
                                        cp_temp(toi)))*((2*w/y**2) + (2/r**2))
        m[i, i+1] = -(k_carbon(toi) /
                      (rho*cp_temp(toi)))*(t/r**2)
        m[i, i-1] = -(k_carbon(toi) /
                      (rho*cp_temp(toi)))*(t/r**2)

    # Boundary points of first-second layer interface, assume no heat transport
    # in r over the small time-step.
    toi = temp_2d[1, 0, j-1]
    m[r_nodes, r_nodes] = 1 + t * \
        (k_carbon(toi) / (rho*cp_temp(toi)))*(2*w/y**2)
    m[r_nodes, 0] = -(2*w**2/(w+1))*t*(k_carbon(toi)/(rho*cp_temp(toi)))/y**2
    m[r_nodes, 2*r_nodes] = -(2*w/(w+1))*t * \
        (k_carbon(toi)/(rho*cp_temp(toi)))/y**2

    toi = temp_2d[1, r_nodes-1, j-1]
    m[2*r_nodes-1, 2*r_nodes-1] = 1 + t * \
        (k_carbon(toi) /
         (rho*cp_temp(toi)))*(2*w/y**2)
    m[2*r_nodes-1, r_nodes-1] = -(2*w**2/(w+1))*t*(k_carbon(
        toi)/(rho*cp_temp(toi)))/y**2
    m[2*r_nodes-1, 3*r_nodes-1] = -(2*w/(w+1))*t*(k_carbon(
        toi)/(rho*cp_temp(toi)))/y**2

    # heat conduction in inner Graphite, regular central difference method
    # Understand that this is for layers between the 2nd layer and the bottom.
    # The y-index on temp_2d starts at 2. The r-index spans 1 to end.
    for i in range(2, y_nodes-1):
        for u in range(1, r_nodes-1):
            toi = temp_2d[i, u, j-1]
            m[i*r_nodes + u, i*r_nodes + u - r_nodes] = - \
                (k_carbon(toi) /
                 (rho*cp_temp(toi)))*(t/y**2)
            m[i*r_nodes + u, i*r_nodes + u + r_nodes] = - \
                (k_carbon(toi) /
                 (rho*cp_temp(toi)))*(t/y**2)
            m[i*r_nodes + u, i*r_nodes + u] = 1 + (k_carbon(toi)/(
                rho*cp_temp(toi)))*t*((2/y**2) + (2/r**2))
            m[i*r_nodes + u, i*r_nodes + u + 1] = - \
                (k_carbon(toi) /
                 (rho*cp_temp(toi)))*(t/r**2)
            m[i*r_nodes + u, i*r_nodes + u - 1] = - \
                (k_carbon(toi) /
                 (rho*cp_temp(toi)))*(t/r**2)

# boundary condition for heat insulation.
    for i in range(2, y_nodes-1):
        toi = temp_2d[i, 0, j-1]
        # left edge condition
        m[i*r_nodes, i*r_nodes - r_nodes] = - \
            (k_carbon(toi) /
             (rho*cp_temp(toi)))*(t/y**2)
        m[i*r_nodes, i*r_nodes + r_nodes] = - \
            (k_carbon(toi) /
             (rho*cp_temp(toi)))*(t/y**2)
        m[i*r_nodes, i*r_nodes] = 1 + (k_carbon(toi)/(
            rho*cp_temp(toi)))*t*((2/y**2) + (2/r**2))
        m[i*r_nodes, i*r_nodes + 1] = -2 * \
            (k_carbon(toi)/(rho*cp_temp(toi)))*(t/r**2)

        # Right Edge Condition
        toi = temp_2d[i, r_nodes-1, j-1]
        m[i*r_nodes + r_nodes - 1, i*r_nodes + r_nodes - 1 - r_nodes] = - \
            (k_carbon(toi) /
             (rho*cp_temp(toi)))*(t/y**2)
        m[i*r_nodes + r_nodes - 1, i*r_nodes + r_nodes - 1 + r_nodes] = - \
            (k_carbon(toi) /
             (rho*cp_temp(toi)))*(t/y**2)
        m[i*r_nodes + r_nodes - 1, i*r_nodes + r_nodes - 1] = 1 + (k_carbon(toi)/(
            rho*cp_temp(toi)))*t*((2/y**2) + (2/r**2))
        m[i*r_nodes + r_nodes - 1, i*r_nodes + r_nodes - 1 - 1] = -2 * \
            (k_carbon(toi)/(rho*cp_temp(toi)))*(t/r**2)

    # heat conduction at the bottom
    for i in range(1, r_nodes-1):
        toi = temp_2d[y_nodes-1, i, j-1]
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i] = 1 + \
            (k_carbon(toi)/(rho*cp_temp(toi)))*t*((2/y**2) + (2/r**2))
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - r_nodes] = -2*(k_carbon(toi) /
                                                                            (rho*cp_temp(toi)))*(t/y**2)
        # There is no "+ r_nodes" because it'll leave the bounds
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - 1] = - \
            (k_carbon(toi) /
             (rho*cp_temp(toi)))*(t/r**2)
        m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i + 1] = - \
            (k_carbon(toi) /
             (rho*cp_temp(toi)))*(t/r**2)

    # Here we are setting the boundary conditions not reached in previous loop.
    i = r_nodes - 1
    toi = temp_2d[y_nodes-1, i, j-1]
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i] = 1 + \
        (k_carbon(toi)/(rho*cp_temp(toi)))*t*((2/y**2) + (2/r**2))
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - r_nodes] = -2 * \
        (k_carbon(toi) /
         (rho*cp_temp(toi)))*(t/y**2)
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - 1] = -2 * \
        (k_carbon(toi) /
         (rho*cp_temp(toi)))*(t/r**2)

    # Recall that "range" is exclusive, so the y_nodes-1*r_nodes indices were
    # not covered.
    i = 0
    toi = temp_2d[y_nodes-1, i, j-1]
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i] = 1 + \
        (k_carbon(toi)/(rho*cp_temp(toi)))*t*((2/y**2) + (2/r**2))
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i - r_nodes] = -2 * \
        (k_carbon(toi) /
         (rho*cp_temp(toi)))*t/y**2
    m[(y_nodes-1)*r_nodes + i, (y_nodes-1)*r_nodes + i + 1] = -2 * \
        (k_carbon(toi) /
         (rho*cp_temp(toi)))*(t/r**2)

    for i in range(r_nodes):
        # All the temperatures in 1-dim for r at a certain time j
        # (big over-arching forLoop). Note this is for calculating Temps at
        # nodes on the FIRST layer (top layer, y_node = 0).
        temp_left[i] = temp_1d[i, j]
        templeft_all[i, j] = temp_1d[i, j]

    for i in range(r_nodes):
        for k in range(1, (y_nodes-1)):
            # Retrieving temperatures from 2nd layer to 2nd-to-last layer.
            temp_left[k*r_nodes + i] = temp_2d[k, i, j-1]
            templeft_all[k*r_nodes + i, j] = temp_2d[k, i, j-1]

    for i in range(r_nodes):
        # Getting temperatures from the last layer in y.
        temp_left[(y_nodes-1)*r_nodes + i] = temp_2d[y_nodes-1, i, j-1]
        templeft_all[(y_nodes-1)*r_nodes + i, j] = temp_2d[y_nodes-1, i, j-1]

    temp_right = np.linalg.solve(m, temp_left)
    for k in range(y_nodes):
        for i in range(r_nodes):
            temp_2d[k, i, j] = temp_right[k*r_nodes + i]


# Calculate heat flux
heatflux = np.zeros((r_nodes, time_frames))
for i in range(r_nodes):
    for j in range(time_frames):
        heatflux[i, j] = (temp_2d[0, i, j] - temp_2d[1, i, j]
                          ) * k_carbon(temp_2d[0, i, j])/(y/w)


# =============================================================================
# ~~~~~ Heat Flux Analysis: Peak finding, flux expansion, lambda-int etc ~~~~~
# =============================================================================

# Function for Simpson's Rule Integration for calculating deposited power and
# integrated heat flux decay length. It may use a combination of Simpson's 1/3
# Rule and the Trapezoidal Rule depending your number of intervals.

def integrate(y_vals, dx):  # y_vals is an array of 1D heatflux & h is stepsize
    if len(y_vals) % 2 == 1:  # Simpson's 1/3 Rule ... all odd numbers
        i = 1
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            if i % 2 == 0:
                total += 2 * y
            else:
                total += 4 * y
            i += 1
        return total * (dx / 3.0)
    elif len(y_vals) % 3 == 1:  # Simpson's 3/8 Rule ... even numbers (ex. 16)
        total = y_vals[0] + y_vals[-1]
        for y in y_vals[1:-1]:
            total += 3 * y
        return total * (3 * dx / 8)
    else:  # Combo of 1/3 & Trapezoidal Rule for points !=(2N+1),(3N+1) e.g. 14
        i = 1
        total = y_vals[0] + y_vals[-2]
        for y in y_vals[1:-2]:
            if i % 2 == 0:
                total += 2 * y
            else:
                total += 4 * y
            i += 1
        total = total * (dx / 3.0)
        total += (dx / 2) * (y_vals[-2] + y_vals[-1])  # Trapezoidal Rule
        return total


fx = np.zeros((36, 2))  # dimensions of flux expansion data
with open(flux_expan) as csv_file:
    csv_reader = csv.reader(csv_file)
    i = 0
    for row in csv_reader:
        fx[i, 0] = float(row[0])  # time in s
        fx[i, 1] = float(row[1])  # flux expansion
        i += 1
# creates an interpolated function for finding flux expansion
fx_interp = interp1d(fx[:, 0], fx[:, 1])
q_bckgrd = 0

p_div = []
q_peaks = []
r_peaks = []

# The following loops makes 1D arrays of q_peak, fx, and p_div for each dt
for j in range(time_frames):
    y_vals = heatflux[:, j]
    pwr = 2*mt.pi*integrate(y_vals, r)
    # Using Simpson's rule to calc deposited power and appending it to a list.
    p_div.append(pwr)
    q_peaks.append(np.max(y_vals))  # find peak heat flux @ time t
    peak_index = np.argmax(y_vals)
    # Store the radius where the peak occurs
    r_peaks.append(radius_m[peak_index])

t_plt = np.array(times[0:time_frames])
t_dim = 2785  # Time step that you want to observe in plots
t1 = 0.4515  # Settting time domain of plots of interest. Uncomment below to
t2 = 0.4535  # enable time domain.
r_plt = np.array(radius_m[0:r_nodes])
r_dim = peak_index + 1  # the radial node for peak heat flux
q_peak = q_peaks[t_dim]  # peak heat flux value
r_peak = r_peaks[t_dim]  # Get the radius of the peak heat flux
a_wet = p_div[t_dim]/q_peak
p_div = np.array(p_div)

# Calculating lambda-int
val_fcn = heatflux[:, t_dim] - q_bckgrd
f_x = fx_interp(t_plt[t_dim])
lambda_int = integrate(val_fcn, r)/(q_peak/f_x)
# %% Figures of Interest

plt.figure(1)
c = plt.pcolormesh(t_plt, r_plt, heatflux/1e6, cmap='turbo', shading='gouraud')
cbar = plt.colorbar(c)
cbar.set_label(r'Heat Flux $\mathbf{(MW/m^2)}$', fontsize=14,
               weight='bold', rotation=270, labelpad=20, y=0.5)
cbar.ax.tick_params(labelsize=14)
# plt.xlim(t1, t2)  # Uncomment to set time domain
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('R (m)', fontsize=14, weight='bold')
plt.xlabel('Time (s)', fontsize=14, weight='bold')
plt.title('Temporal Heat Flux Evolution', weight='bold')


plt.figure(2)
c = plt.pcolormesh(
    t_plt, r_plt, temp_2d[0, :, :], cmap='turbo', shading='gouraud')
cbar = plt.colorbar(c)
cbar.set_label('Temperature ($^\circ$C)', fontsize=14,
               weight='bold', rotation=270, labelpad=20, y=0.5)
cbar.ax.tick_params(labelsize=14)
plt.xlim(t1, t2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('R (m)', fontsize=14, weight='bold')
plt.xlabel('Time (s)', fontsize=14, weight='bold')
plt.title('Surface Temperature Profile', weight='bold')

plt.figure(3)
plt.plot(t_plt, heatflux[r_dim]/1e6, 'k--')
plt.title(
    'Radial Heat Flux Profile @ R ={:.4f}m'.format(radius_m[r_dim]), weight='bold')
plt.xlabel('Time (s)', fontsize=14, weight='bold')
plt.ylabel(r'Heat Flux $\mathbf{(MW/m^2)}$', fontsize=14, weight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left')

# Radial Heat Flux Profile; you may need to adjust the positions of the text on
# the figure depending on your profile.
plt.figure(4)
plt.plot(radius_m, heatflux[:, t_dim], 'r')
plt.title(
    'Peak Heat Flux @ t={:.3f}s'.format(t_plt[t_dim]), fontsize=14, weight='bold')
plt.text(0.80, .86*q_peak, 'P_div = {:.2f} MW'.format(p_div[t_dim]/1e6))
plt.text(0.80, .77*q_peak, 'A_wet = {:.2f} $m^2$'.format(a_wet))
plt.text(0.80, .687*q_peak, '$\lambda$_int = {:.3f} m'.format(lambda_int))
plt.text(0.80, .6*q_peak, '$f_x$ = {:.2f}'.format(f_x))
plt.annotate('q_peak =\n {:.1f} MW/$m^2$'.format(q_peak/1e6), xy=(r_peak, q_peak), xytext=(.88 *
             r_peak, .75*q_peak), arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=4))
plt.xlabel('Radius (m)')
plt.ylabel('Heat Flux $(W/m^2)$', fontsize=14, weight='bold')


fig5, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t_plt, heatflux[r_dim], 'k--', label='Heat Flux')
ax2.plot(t_plt, p_div/1e6, 'g--', label='Deposited Power')
plt.title('Profile @ R ={:.4f}m'.format(radius_m[r_dim]), weight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Heat Flux $(W/m^2)$', fontsize=14, weight='bold')
ax2.set_ylabel('Deposited Power (MW)', color='g', fontsize=14, weight='bold')
ax1.ticklabel_format(axis='y', scilimits=(6, 6))
plt.xlim(t1, t2)
plt.locator_params(axis='x', nbins=5)
