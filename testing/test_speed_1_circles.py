# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 22:09:58 2016

@author: rcroke
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import time

## about 30 plots/second
#fig, ax = plt.subplots()
#line, = ax.plot(np.random.randn(100))
#
#tstart = time.time()
#num_plots = 0
#while time.time()-tstart < 1:
#    line.set_ydata(np.random.randn(100))
#    plt.pause(0.001)
#    num_plots += 1
#print(num_plots)


## Around 40 fps
#fig, ax = plt.subplots()
#line, = ax.plot(np.random.randn(100))
#
#tstart = time.time()
#num_plots = 0
#while time.time()-tstart < 1:
#    line.set_ydata(np.random.randn(100))
#    fig.canvas.draw()
#    fig.canvas.flush_events()
#    num_plots += 1
#print(num_plots)

fig, ax = plt.subplots()
line, = ax.plot(np.random.randn(100))
plt.show(block=False)

tstart = time.time()
num_plots = 0
while time.time()-tstart < 5:
    line.set_ydata(np.random.randn(100))
    ax.draw_artist(ax.patch)
    ax.draw_artist(line)
    fig.canvas.blit(ax.bbox)
    #fig.canvas.update()
    fig.canvas.flush_events()
    num_plots += 1
print(num_plots/5)