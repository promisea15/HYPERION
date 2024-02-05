# HYPERION
 2-D Finite Difference Heat Transfer Code

HYPERION is able to estimate the surface heat flux, deposited power, and integrated heat flux decay width in fusion reactors. 
Users can use 'HYPERION_buildmesh' or 'HYPERION_main'. The buildmesh script is for users that do not have set time and position data. 
To calculate heat flux users must input surface temperature data as a .csv file [temp_file]. In HYPERION_main time and position files [time_file, position_file] must also be in .csv file format. These .csv files are parsed and are saved as numpy arrays. 

The heat flux calculation uses an implicit method and it is assumed that the bottom and side surfaces are insulated. Temperature dependent or constant thermal material properties may be used in the calculation. For temperature dependent properties it is best to write functions like ['k_carbon'] and ['cp_temp'] which can be found in ['temp_dep_prop.py'].

The function ['integrate'] uses Simpson's Rule and a combination of the Trapezoidal Rule and Simpson's Rule to calculate deposited power.
Users can also include flux expansion data which can be used to calculate integrated heat flux decay width. Note that the flux expansion data may be provided at timesteps that do not align with your own temperature data, therefore the 'interp1d' function is used to interpolate the flux expansion at different time steps. 
Note that this has been removed from HYPERION_main for the time being. 

The heat transmission coefficient or [alpha] parameter is used to characterize the thermal properties of the surface layers that form on the divertor/limiter during reactor operation. In HYPERION [alpha] appears in the coefficient matrix [m] in the first layer section. Note that the 1st-2nd layer interface, or the interface between nodes 1 & 2 in the depth direction, is much thinnner than that of the bulk. This is to resolve more accurate temperature values for the heat flux calculation. 
Note that users have the ability to make their mesh as thin as they want, but it could come with heavy computational cost if it is made too thin. Obviously, a thinner mesh provides a more accurate temperature value. It is suggested that only the top layer thickness is adjusted via the w-value.
