
#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import math
from cvxpy import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')
#local imports
from line.funcs import *
from triangle.funcs import *
from params import *
#if using termux
import subprocess
import shlex
#end if

#Line parameters
x1 = np.array([1,-1,0])
x2 = np.array([-1,2,2])
m1 = np.array([2,3,1])
m2 = np.array([5,1,0])
M = np.block([[m1],[m2]]).T
x2minusx1 = x2-x1
matA = np.block([[m1],[m2], [x2minusx1]])

mat_rank = LA.matrix_rank(matA)
print("Rank=", mat_rank)
if(mat_rank == 3): # Lines don't intersect
    mtm = M.T@M
    b = M.T@x2minusx1 
    lamda = LA.solve(mtm,b)
    A = x1 + lamda[0]*m1
    B = x2 - lamda[1]*m2
    dist = LA.norm(A-B)
    print("A",A)
    print("B",B)
    print("B-A",B-A)
    print("Distance=", dist)
else:
    print("Lines intersect each other")

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
