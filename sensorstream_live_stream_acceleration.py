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
import scipy.signal as signal
import math
import time
import numpy as np


##### H E L P E R   FUNCTIONS
#####

def butter_pass(cutoff, fs, order=5,btype='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_pass_filter(data, cutoff, fs, order=5,btype='low'):
    b, a = butter_pass(cutoff, fs, order=order,btype='low')
    y = lfilter(b, a, data)
    return y

def marcs_processing_LPF(data):
    # Filter requirements.
    order = 2
    nsamps = len(data)
    N = order
    fs = 25.0       # sample rate, Hz
    cutoff = 8.0  # desired cutoff frequency of the filter, Hz

    ############### Apply 5Hz LPF
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_pass(cutoff, fs, order)

    # Prepare data for filter.
    data = np.array(data)

    # Apply the filter
    y = butter_pass_filter(data, cutoff, fs, order)
    
#    ############### DOWNSAMPLE
#    # downsampling using stock methods
#    secs   = len(y)/fs  # Number of seconds in signal y
#    samps  = secs*60.0     # Number of samples to downsample
#    y_down = signal.resample(y, samps)
#    
#    ############### Apply 3Hz LPF
#    # Filter requirements.
#    order = 2
#    nsamps = len(y_down)
#    N = order
#    fs = 60.0       # down sample rate, Hz
#    cutoff = 3  # desired cutoff frequency of the filter, Hz
#
#    # Get the filter coefficients so we can check its frequency response.
#    b, a = butter_pass(cutoff, fs, order)
#    
#    # Prepare data for filter.
#    data_down = np.array(y_down)
#
#    # Apply the filter
#    y_3Hz_LPF = butter_pass_filter(data_down, cutoff, fs, order)
    
    return y

def marcs_processing_HPF(data):
    # Filter requirements.
    order = 2
    nsamps = len(data)
    N = order
    fs = 60.0       # sample rate, Hz
    cutoff = 0.5  # desired cutoff frequency of the filter, Hz

    ############### Apply 5Hz LPF
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_pass(cutoff, fs, order,btype='highpass')

    # Prepare data for filter.
    data = np.array(data)

    # Apply the filter
    y = butter_pass_filter(data, cutoff, fs, order,btype='highpass')
    
    return y

def pitch_calc(data):
    # return in degrees
    return np.arctan(data[0]/np.sqrt(data[1]**2 + data[2]**2))
def roll_calc(data):
    # return in degrees
    return np.arctan(data[1]/np.sqrt(data[0]**2 + data[2]**2))
def nutation_calc(pitch,roll):
    return np.arccos(np.cos(pitch)*np.cos(roll))
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

fig, ax = plt.subplots(figsize=(16,10))
#line, = ax.plot(np.random.randn(100))

#fig = plt.figure(figsize=(16,10))

# raw data
ax = plt.subplot("311")
ax.set_xlim(0, 50)
ax.set_ylim(-2, 2)
ax.set_title("raw acceleration data")
ax.set_ylabel("g$/s^2$",fontsize=18)
ax.hold(True)

line  = ax.plot(Ax,label='Acc x',c="green")[0]
line2 = ax.plot(Ay,label='Acc y',c="black")[0]
line3 = ax.plot(Az,label='Acc z',c="red")[0]
legend = ax.legend(loc='lower left', shadow=True)

# filtered data
ax2 = plt.subplot("312")
ax2.set_xlim(0, 50)
ax2.set_ylim(-2, 2)
ax2.set_title("filtered acceleration data")
ax2.set_ylabel("g$/s^2$",fontsize=18)
ax2.hold(True)

f_line  = ax2.plot(Ax,label='Acc x',c="green")[0]
f_line2 = ax2.plot(Ay,label='Acc y',c="black")[0]
f_line3 = ax2.plot(Az,label='Acc z',c="red")[0]

# test plot
ax3 = plt.subplot("313")
test = 1 # 0 for G, 1 for tilt angle
if (test == 0):
    ax3.axhline(y=1.2, xmin=0, xmax=1,c="green",linewidth=0.5,zorder=0,label='Felt during airplane banking')
    ax3.axhline(y=3.0, xmin=0, xmax=1,c="red",linewidth=0.5,zorder=0,label='Max rollercoaster')
    ax3.set_ylim([0,2*np.sqrt(3.1)])
    ax3.set_title("G forces")
    ax3.set_ylabel("g/$s^2$",fontsize=18)
    legend = ax3.legend(loc='lower left', shadow=True)
    t_line = ax3.plot(G)[0]
else:
    ax3.set_ylim([-180,180])
    ax3.set_title("nutation, pitch and roll angles")
    ax3.set_ylabel("degrees",fontsize=18)
    ax3.axhline(y=0.0, xmin=0, xmax=1,c="black",linewidth=0.5,zorder=0,label='zero degrees')
    pitch_line    = ax3.plot(G,label='pitch',c='green')[0]
    roll_line     = ax3.plot(G,label='roll',c='red')[0]
    nutation_line = ax3.plot(G,label='nutation',c='yellow')[0]
    legend = ax3.legend(loc='lower left', shadow=True)

fig.suptitle('Three-axis accelerometer streamed from Sensorstream',fontsize=18)
plt.show(block=False)
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
    
    marc_acc_x = marcs_processing_LPF(Ax)
    marc_acc_y = marcs_processing_LPF(Ay)
    marc_acc_z = marcs_processing_LPF(Az)
    
    num_samples = len(marc_acc_z)
    f_s_marc = 60.0 # sampling frequency, in Hz
    dt = 1 / f_s_marc # sampling frequency in seconds
    k = np.arange(num_samples)

    marc_time = [k*dt for k in range(0,num_samples)]
    # filter
    #acc_x_savgol = savgol_filter(Ax, window_length=5, polyorder=3)
    #acc_y_savgol = savgol_filter(Ay, window_length=5, polyorder=3)
    #acc_z_savgol = savgol_filter(Az, window_length=5, polyorder=3)
    
    data = np.vstack([Ay,Ax,Az]) # need to get right axes in right places
    pitch    = pitch_calc(data)
    roll     = roll_calc(data)
    nutation = nutation_calc(pitch,roll)*180.0/np.pi
    pitch = pitch*180.0/np.pi
    roll = roll*180.0/np.pi
#    tilt_angles = []
#    for i,val in enumerate(G): 
#        if (val == 0):
#            tilt_angles.append(0.0)
#        elif (math.isnan(np.arccos(Az[i]/val))):
#            tilt_angles.append(0.0)
#        else:
#            tilt_angles.append(np.arccos(Az[i]/val) * (180.0 / math.pi))
            
    print(Ax[0],Ay[1],Az[2])   
    
    line.set_xdata(x)
    line.set_ydata(Ax)
    line2.set_xdata(x)
    line2.set_ydata(Ay)
    line3.set_xdata(x)
    line3.set_ydata(Az)
    #ax.set_xlim(count, count+50)
    
    f_line.set_xdata(x)
    f_line.set_ydata(marc_acc_x)
    f_line2.set_xdata(x)
    f_line2.set_ydata(marc_acc_y)
    f_line3.set_xdata(x)
    f_line3.set_ydata(marc_acc_z)
    #ax2.set_xlim(count, count+50)

    if (test == 0):
        t_line.set_xdata(x)
        t_line.set_ydata(G)
    else:
        pitch_line.set_xdata(x)
        pitch_line.set_ydata(pitch)
        roll_line.set_xdata(x)
        roll_line.set_ydata(roll)
        nutation_line.set_xdata(x)
        nutation_line.set_ydata(nutation)
    
    # Drawing three graphs really slows it down. 
    # if you do one axis and one graph it can handle 
    # 200 - 300 Hz data easy
    
    
    ax.draw_artist(ax.patch)
    ax.draw_artist(ax2.patch)
    ax.draw_artist(ax3.patch)

    # redraw just the points
    ax.draw_artist(line)
    ax.draw_artist(line2)
    ax.draw_artist(line3)
    ax2.draw_artist(f_line)
    ax2.draw_artist(f_line2)
    ax2.draw_artist(f_line3)
    if (test == 0):
        ax3.draw_artist(t_line)
    else:
        ax3.draw_artist(pitch_line)
        ax3.draw_artist(roll_line)
        ax3.draw_artist(nutation_line)
    
    fig.canvas.blit(ax.bbox)
    fig.canvas.blit(ax2.bbox)
    fig.canvas.blit(ax3.bbox)
    #fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()
    count+=1
    #x = np.arange(count,count+50,1)
    
    # tops out at about 25 fps :|
    print "FPS: ",1.0/(time.time() - tstart)