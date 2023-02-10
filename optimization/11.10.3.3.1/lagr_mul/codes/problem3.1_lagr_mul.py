#Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
import os
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')
#local imports
from line.funcs import *
from conics.funcs import *
from triangle.funcs import *
from params import *
#if using termux
import subprocess
import shlex

#Input parameters and other points
iters = 1000 
alpha = 0.05
x_old = np.array([[0],[8/math.sqrt(3)]])
O = np.array([[0],[0]])
T = np.array([[-8],[0]])
S = np.array([[4],[12/math.sqrt(3)]])
eps = 1e-6
n = np.array([[1],[-math.sqrt(3)]])
m = np.array([[1],[1/math.sqrt(3)]])
c = -8
i = 0
L = line_gen(S,T)
plt.plot(L[0],L[1])
dot_p = 1 
#Function to plot the circle
def plot_circ(p):
    C = circ_gen(O.T,np.linalg.norm(O-p))
    plt.plot(C[0],C[1])

#Iterate until the dot product is zero or the iterations are reached
while(i<iters and dot_p != 0):
    dot_p = m.T@(O-x_old)
    delta = alpha*m*dot_p
    #Every 25 iterations, we plot the results
    if i%5 == 0: 
        plot_circ(x_old)
    i=i+1
    x_old = x_old + delta
    #Standard round() is not giving right output. Hence this workaround was used
    dec, integ = np.modf(dot_p)
    dot_p = np.abs(integ+np.round(dec,5))

M = line_gen(O,x_old)
plt.plot(M[0],M[1])
plt.plot(O[0],O[1],'k.')
plt.text(O[0]-1e-2,O[1]+1e-2,'O')
plt.plot(x_old[0],x_old[1],'k.')
plt.text(x_old[0],x_old[1]+1e-2,'P')
plt.grid()
plt.axis('equal')
plt.title('Perpendicular from Origin')
plt.savefig('../figs/problem3.1.pdf')
#else
plt.show()
