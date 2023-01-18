#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
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

def affine_transform(P,c,x):
    return P@x + c

#Input parameters
V = np.array([[9,0],[0,25]])
u = np.array(([0,0]))
f = -225


#Input parameters
a=5
b=3
O=np.array([0,0])

#Vertices of ellipse
G = np.array([a,0])
H = np.array([-a,0])

I = np.array([0,b])
J = np.array([0,-b])

lamda,P = LA.eigh(V)
if(lamda[1] == 0):      # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)
e = np.sqrt(1- lamda[0]/lamda[1])
e_square = e**2
print("Eccentricity of Ellipse is ", e)

n = np.sqrt(lamda[1])*P[:,0]
c1 = (e*(u.T@n) +np.sqrt( e_square*(u.T@n)**2-lamda[1]*(e_square-1)*(LA.norm(u)**2-lamda[1]*f)))/(lamda[1]*e*(e_square-1))
c2 = (e*(u.T@n) -np.sqrt( e_square*(u.T@n)**2-lamda[1]*(e_square-1)*(LA.norm(u)**2-lamda[1]*f)))/(lamda[1]*e*(e_square-1))

F1 = (c1*e_square*n - u)/lamda[1]
F2 = (c2*e_square*n - u)/lamda[1]

fl1 = LA.norm(F1)
fl2 = LA.norm(F2)

m = omat@n
#Points for Ellipse Major Axis
ellipAxis_A = n
ellipAxis_B = -n

#Points for Ellipse Minor Axis
ellipMinorAxis_A = 0.6*m 
ellipMinorAxis_B = -0.6*m

#Generating the ellipse
elli=  ellipse_gen(a,b) 

# Generating lines 
x_AB = line_gen(ellipAxis_A, ellipAxis_B)
x_minor_AB = line_gen(ellipMinorAxis_A, ellipMinorAxis_B)

#Plotting the ellipse
plt.plot(elli[0,:],elli[1,:],label='$Ellipse$')

leg_label = "{} {}".format("Major", "Axis")
plt.plot(x_AB[0,:],x_AB[1,:] ,label = leg_label)

leg_label = "{} {}".format("Minor", "Axis")
plt.plot(x_minor_AB[0,:],x_minor_AB[1,:] ,label = leg_label)

#Labeling the coordinates
plot_coords = np.vstack((F1,F2,H,G)).T
vert_labels = ['$F_1$','$F_2$','$V_1$','$V_2$']
for i, txt in enumerate(vert_labels):
    if ( i == 0) :
      label = "{}".format('$F_1 - Focus 1$' )
    elif ( i == 1) :
      label = "{}".format('$F_2 - Focus 2$' )
    elif ( i == 2) :
      label = "{}".format('$V_1 - Vertex 1$' )
    else :
      label = "{}".format('$V_2 - Vertex 2$' )

    plt.scatter(plot_coords[0,i], plot_coords[1,i], s=15, label = label)
    plt.annotate(txt, # this is the text
                (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(8,5), # distance from text to points (x,y)
                 fontsize=7,
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.gca().legend(loc='lower left', prop={'size':6},bbox_to_anchor=(0.93,0.6))
plt.grid() # minor

plt.axis('equal')
plt.title('Ellipse')
#if using termux
plt.savefig('../figs/problem7.pdf')
#else
plt.show()
