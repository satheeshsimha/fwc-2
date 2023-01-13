#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import cvxpy  as cp

#if using termux
import subprocess
import shlex
#end if


# objective function coeffs
c = np.array(([3, 4]))
x = cp.Variable([2,1])

#Cost function
cost_func = c@x
obj = cp.Maximize(cost_func)

#Constraints
A = np.array(([1,4],[-1,0],[0,-1]))
A_b = np.array(([4,0,0])).reshape(3,1)
constraint = [ A@x <= A_b]


#solution
prob = cp.Problem(obj, constraint)
prob.solve()
print("optimal value:", cost_func.value)
print("optimal var:", x.value.T)

#Drawing lines
x1 = np.linspace(0,5,400)#points on the x axis
y1 = (4-x1)/4
y2 = np.zeros(len(x1))

plt.plot(x1,y1,label = '$x+4y=4$')
plt.plot(x1,y2,label = '$y=0$')
plt.plot(y2,x1,label = '$x=0$')
plt.grid()
plt.xlabel('$x-Axis$')
plt.ylabel('$y-Axis$')
plt.title('Linear Programming')
plt.ylim(-0.5,1.5)

txt = "{} {}".format("Optimum ", "Point") 
plt.plot(x.value[0],x.value[1],color=(1,0,1),marker='o',label= txt)
plt.legend(loc='best')
#if using termux
plt.savefig('../figs/problem1.pdf')
#else
plt.show()
