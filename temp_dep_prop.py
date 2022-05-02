# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:36:12 2021
This function gives the tempertature dependepent thermal conductivity of graphite by taking the surface temperature as input. This function is adapted from KGan (UTK). 
@author: promi
"""
def k_carbon(temp):
    if temp <= 0:
        k = 94.85 #W/m-K
    elif temp > 0 and temp <= 200:
        k = 94.85-(94.85-80.12)*temp/200
    elif temp > 200 and temp <= 1000:
        k = 80.12-(80.12-42.64)*(temp-200)/800
    elif temp > 1000 and temp < 2000:
        k = 42.46-(42.64-29.31)*(temp-1000)/1000
    elif temp >= 2000 and temp <= 2600:
        k = 29.31-(29.31-27.21)*(temp-2000)/600
    elif temp > 2600 and temp <= 2900:
        k = 27.21-(27.21-24.56)*(temp-2600)/300
    elif temp > 2900 and temp <= 3500: 
        k = 24.56-(24.56-10.05)*(temp-2900)/600
    elif temp > 3500 and temp <= 3600:
        k = 10.05-(10.05-5.87)*(temp-3500)/100
    else:
        k = 5.87
    return k

def cp_temp(i):
    if i <= 0:
        cp = 551 #Specific Heat Capacity (J/kg-K)
    elif i > 0 and i <= 20:
        cp = 551 + (713-551)*i/20
    elif i > 20 and i <= 100:
        cp = 713 + (1125-713)*(i-20)/80
    elif i > 100 and i <= 200:
        cp = 1125 + (1383-1125)*(i-100)/100
    elif i > 200 and i <= 300:
        cp = 1383 + (1525-1383)*(i-200)/100
    elif i > 300 and i <= 400:
        cp = 1525 + (1614.-1525.)*(i-300)/100
    elif i > 400 and i <= 500:
        cp = 1614 + (1675.-1614.)*(i-400)/100 
    elif i > 500 and i <= 3600: 
        cp = 1675 + (2238-1675)*(i-500)/3100
    else:
        cp = 2238
    return cp
