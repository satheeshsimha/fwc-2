#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from numpy import linalg as LA
from math import *
import matplotlib.cm as cm
import matplotlib.legend as Legend

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import ellipse_gen
from params import *
#if using termux
import subprocess
import shlex
#end if


#Input parameters
a=2  #Radius of cone
b= 0.5
O=np.array([0,0])
l = 3*a # Slant height
h = np.sqrt(l**2 - a**2) #Height of cone

#Base of Cone drawn as Ellipse 
A = np.array([a,0])
B = np.array([-a,0])

C = np.array([0,h]) #Apex of cone

#Generating the ellipse
elli=  ellipse_gen(a,b) 

# Generating lines 
x_AB = line_gen(A,B)
x_OC = line_gen(O,C)
x_AC = line_gen(A,C)
x_BC = line_gen(B,C)

#Plotting the ellipse
plt.plot(elli[0,:],elli[1,:])

#Plotting lines
plt.plot(x_AB[0,:],x_AB[1,:] )
plt.plot(x_OC[0,:],x_OC[1,:] )
plt.plot(x_AC[0,:],x_AC[1,:] )
plt.plot(x_BC[0,:],x_BC[1,:] )

#patches.Arc(C,1,0.5, asin(1/3), theta2= asin(1/3))
#Labeling the coordinates
plot_coords = np.vstack((A,B,C,O)).T
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.scatter(plot_coords[0,i], plot_coords[1,i])
    plt.annotate(txt, # this is the text
                (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(8,0), # distance from text to points (x,y)
                 fontsize=7,
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')

#plt.gca().legend(loc='lower left', prop={'size':6},bbox_to_anchor=(0.93,0.6))
plt.grid() # minor

#plt.axis('equal')
plt.title('Geometric Programming')
#if using termux
plt.savefig('../figs/problem26.pdf')
#else
plt.show()
