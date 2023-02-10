#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')

#local imports
#from CoordGeo.line.funcs import *
#from triangle.funcs import *
from conics.funcs import parab_gen
from line.funcs import line_gen
from params import *
#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c

#Input parameters
V = np.array([[1,0],[0,0]])
u = np.array(([0,1]))
f = -3 
R = np.array([[0,1],[-1,0]]) # Rotation matrix
q = np.array(([1,1])) #Given point 

m = R@(V@q+u) #Slope of Tangent 
n = (V@q+u) # Slope of Normal
c = m.T@q

#Generating parabola
num_points = 100
x = np.linspace(-5,5,num_points)
y = 0.5*(3-x**2)   # y = 0.5*(3-x^2) 

# Generating normal  
C = np.array(([c,0]))
p = n*5 + C 
x_pq = line_gen(p,q)

# Generating tangent  
d = -(q.T@V@q + u.T@q) #Intercept for Tangent
D= np.array(([-d,0]))
tangent_A = 5*m+D  
tangent_B = -5*m+D
tangent_AB = line_gen(tangent_A, tangent_B)

#Plotting all shapes
plt.plot(x, y,label ='$Parabola$')
plt.plot(x_pq[0,:],x_pq[1,:] ,label = 'x-y=0')
plt.plot(tangent_AB[0,:],tangent_AB[1,:] ,label = 'Tangent')


#Labeling the coordinates
label = "{}({:.0f},{:.0f})".format('q', q[0],q[1]) #Form label as A(x,y)
plt.annotate(label, # this is the text
            (q[0], q[1]), # this is the point to label
            textcoords="offset points", # how to position the text
            xytext=(20,0), # distance from text to points (x,y)
            ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best', fontsize = 'small')
plt.grid() # minor
plt.axis('equal')
plt.title('Normal to Parabola')
#if using termux
plt.savefig('../figs/problem22.pdf')
plt.show()
