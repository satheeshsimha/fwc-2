#Least Squares using SVD

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D

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

#Line parameters
x1 = np.array([1,-1,0])
ax1 = x1.reshape(3,-1)
x2 = np.array([-1,2,2])
ax2 = x2.reshape(3,-1)
m1 = np.array([2,3,1])
m2 = np.array([5,1,0])

k = ax2-ax1

M = np.block([[m1],[m2]]).T

numberOfColumns = np.shape(M)[1]
numberOfRows = np.shape(M)[0]

#Performing SVD
U, D, VT = LA.svd(M)

sz = np.shape(D)[0]

#If rank is less than number of rows or cols
#Pad additional columns and rows to the diagonal matrix with 0s
if (sz < numberOfRows or sz < numberOfColumns): 
    diag = np.diag(D) 
    Sigma = np.zeros((numberOfRows ,numberOfColumns ))
    Sigma[:sz,:sz] = diag 

V = VT.T
lamda_n = np.zeros((1,numberOfColumns))
#Compute lamda min value
for i in range(sz) :
    lamda_n += ((U[:,i].T@(-k))/D[i])*V[i,:] 
   
print(lamda_n)

#Optimal points on each line
A = x1-lamda_n[:,0]*m1
B = x2+lamda_n[:,1]*m2

print("U", U)
print("V", V)
print("Sigma", Sigma)
print("Lambda values = ", lamda_n[0])
print("Distance=", LA.norm(A-B))
print("Point on line 1",A) 
print("Point on line 2",B)

#Points for plotting Line1
line1_point1 = x1 -1*(m1)
line1_point2 = x1 +5*(m1)
line1 = line_gen(line1_point1, line1_point2) 
#Points for plotting Line2
line2_point1 = x2 -1*(m2)
line2_point2 = x2 +5*(m2)
line2 = line_gen(line2_point1, line2_point2) 
#Points for plotting shortest distance between Line 1 and 2 
lineAB = line_gen(A, B) 

#Plotting lines
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(line1[0,:],line1[1,:], line1[2,:],label = 'Line1')
ax.plot3D(line2[0,:],line2[1,:], line2[2,:],label = 'Line2')
ax.plot3D(lineAB[0,:],lineAB[1,:], lineAB[2,:],label = 'AB')

ax.text(A[0],A[1],A[2],'A')
ax.text(B[0],B[1],B[2],'B')
plt.legend(['L1','L2','Normal'])

ax.view_init(60,30)
plt.grid()
plt.tight_layout()

plt.savefig('../figs/problem30.pdf')
plt.show()
