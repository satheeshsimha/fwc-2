
#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')

#local imports
#from conics.funcs import circ_gen
from line.funcs import line_gen

#if using termux
import subprocess
import shlex
#end if

#Plotting (2x-1)**2 +3 
x = np.linspace(-0.5,1.5,40)#points on the x axis
f= (2*x-1)**2 + 3 #Objective function
plt.plot(x,f,color=(1,0,1), label = '$f(x)= (2x-1)^2 + 3$')
plt.grid()
plt.xlabel('$x-Axis$')
plt.ylabel('$y-Axis$')
plt.title('Convexity')

#Convexity/Concavity
a = 0
b = 1.25
lamda = 0.45

c = lamda *a + (1-lamda)*b

f_a = (2*a-1)**2 +3 #f(a)
f_b = (2*b-1)**2 +3 #f(a)

f_c = (2*c-1)**2 +3 #f(c)

f_c_hat = lamda *f_a + (1-lamda)*f_b

#Plot commands
plt.plot([a,a],[0,f_a],color=(1,0,0),marker='o',label="$f(a)$")
plt.plot([b,b],[0,f_b],color=(0,1,0),marker='o',label="$f(b)$")
plt.plot([c,c],[0,f_c],color=(0,0,1),marker='o',label="$f(\lambda a + (1-\lambda)b)$")
plt.plot([c,c],[0,f_c_hat],color=(1/2,2/3,3/4),marker='o',label="$\lambda f(a) + (1-\lambda)f(b)$")
plt.plot([a,b],[f_a,f_b],color=(0,1,1))
#plt.legend(loc='best')
plt.gca().legend(loc='lower left', prop={'size':7},bbox_to_anchor=(0.85,0.4))
#subprocess.run(shlex.split("termux-open ../figs/1.1.pdf"))
#if using termux
plt.savefig('../figs/problem1.pdf')
#else
plt.show()
