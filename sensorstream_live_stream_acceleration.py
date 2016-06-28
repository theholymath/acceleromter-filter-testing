# -*- coding: utf-8 -*-
"""
Spyder Editor

Author:  Ryan Croke
Website: www.TheHolyMath.com

Use:     This script uses the Android app: Sensorstream IMU+GPS 
(https://play.google.com/store/apps/details?id=de.lorenz_fenster.sensorstreamgps&hl=en)
and plots a live stream of the accelerometer data. The flag test = 0 on line 77 
allows the user to either look at the G-forces or the tilt angles. Feel free to 
improve and let me know if you find a speed up for the plotting.
"""



import socket
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter,iirfilter,savgol_filter
import math
import time
import numpy as np


### make connection with phone 
host = ''
port = 5555
 
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))


### Get data ready
# lists for plotting
Ax = [0.0] * 50
Ay = [0.0] * 50
Az = [0.0] * 50

# G-force of 1.2 is what is felt in an elevator. Maybe use this as a simple
# threshold. 
G  = [0.0] * 50
x = [i for i in range(len(Ax))]
 
#used for debugging
 
fig = plt.figure(figsize=(16,10))

# raw data
ax = plt.subplot("311")
ax.set_xlim(0, 50)
ax.set_ylim(-2, 2)
ax.set_title("raw acceleration data")
ax.set_ylabel("g$/s^2$",fontsize=18)
ax.hold(True)

line  = ax.plot(Ax,label='Acc x')[0]
line2 = ax.plot(Ay,label='Acc y')[0]
line3 = ax.plot(Az,label='Acc z')[0]
legend = ax.legend(loc='lower left', shadow=True)

# filtered data
ax2 = plt.subplot("312")
ax2.set_xlim(0, 50)
ax2.set_ylim(-2, 2)
ax2.set_title("filtered acceleration data")
ax2.set_ylabel("g$/s^2$",fontsize=18)
ax2.hold(True)

f_line  = ax2.plot(Ax,label='Acc x')[0]
f_line2 = ax2.plot(Ay,label='Acc y')[0]
f_line3 = ax2.plot(Az,label='Acc z')[0]

# test plot
ax3 = plt.subplot("313")
test = 0 # 0 for G, 1 for tilt angle
if (test == 0):
    ax3.axhline(y=1.2, xmin=0, xmax=1,c="green",linewidth=0.5,zorder=0,label='Felt during airplane banking')
    ax3.axhline(y=3.0, xmin=0, xmax=1,c="red",linewidth=0.5,zorder=0,label='Max rollercoaster')
    ax3.set_ylim([0,2*np.sqrt(3.1)])
    ax3.set_title("G forces")
    ax3.set_ylabel("g/$s^2$",fontsize=18)
    legend = ax3.legend(loc='lower left', shadow=True)
else:
    ax3.set_ylim([-180,180])
    ax3.set_title("Tilt Angles")
    ax3.set_ylabel("degrees",fontsize=18)
t_line = ax3.plot(G)[0]

fig.suptitle('Three-axis accelerometer streamed from Sensorstream',fontsize=18)
plt.show(False)
plt.draw()

# cache the background
background = fig.canvas.copy_from_bbox(fig.bbox)

count = 0 
print("Success binding")
while 1:
    # time it
    tstart = time.time()
    message, address = s.recvfrom(8192)
    messageString = message.decode("utf-8")
    Acc = messageString.split(',')[2:5]
    Acc = [float(Acc[i])/10.0 for i in range(3)]
    
    # appending and deleting is order 10e-5 sec
    Ax.append(Acc[0])
    del Ax[0]
    Ay.append(Acc[1])
    del Ay[0]
    Az.append(Acc[2])
    del Az[0]
    G.append(np.sqrt(Ax[-1]**2 + Ay[-1]**2 + Az[-1]**2))
    del G[0]
    
    # filter
    acc_x_savgol = savgol_filter(Ax, window_length=5, polyorder=3)
    acc_y_savgol = savgol_filter(Ay, window_length=5, polyorder=3)
    acc_z_savgol = savgol_filter(Az, window_length=5, polyorder=3)
    
    tilt_angles = []
    for i,val in enumerate(G): 
        if (val == 0):
            tilt_angles.append(0.0)
        elif (math.isnan(np.arccos(Az[i]/val))):
            tilt_angles.append(0.0)
        else:
            tilt_angles.append(np.arccos(Az[i]/val) * (180.0 / math.pi))
            
    print(Ax[0],Ay[1],Az[2])   
    
    line.set_xdata(x)
    line.set_ydata(Ax)
    line2.set_xdata(x)
    line2.set_ydata(Ay)
    line3.set_xdata(x)
    line3.set_ydata(Az)
    #ax.set_xlim(count, count+50)
    
    f_line.set_xdata(x)
    f_line.set_ydata(acc_x_savgol)
    f_line2.set_xdata(x)
    f_line2.set_ydata(acc_y_savgol)
    f_line3.set_xdata(x)
    f_line3.set_ydata(acc_z_savgol)
    #ax2.set_xlim(count, count+50)

    if (test == 0):
        t_line.set_xdata(x)
        t_line.set_ydata(G)
    else:
        t_line.set_xdata(x)
        t_line.set_ydata(tilt_angles)
    #ax3.set_xlim(count, count+50)
    
    # restore background
    fig.canvas.restore_region(background)

    # redraw just the points
    ax.draw_artist(line)
    ax.draw_artist(line2)
    ax.draw_artist(line3)
    ax2.draw_artist(f_line)
    ax2.draw_artist(f_line2)
    ax2.draw_artist(f_line3)
    ax3.draw_artist(t_line)

    # fill in the axes rectangle
    fig.canvas.blit(fig.bbox)
    
    count+=1
    #x = np.arange(count,count+50,1)
    
    # tops out at about 25 fps :|
    print "FPS: ",1.0/(time.time() - tstart)