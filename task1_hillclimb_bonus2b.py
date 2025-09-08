import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from genetic_all import *


x_min, x_max = -800, 800
y_min, y_max = -800, 800
z_min, z_max=-800,800
d = 40 


xs = np.linspace(x_min, x_max, 100)
ys = np.linspace(y_min, y_max, 100)
zs=np.linspace(z_min, z_max, 100)
X, Y, Z = np.meshgrid(xs, ys,zs)

P = np.zeros_like(X)
for i in range(100): # go by row (Y)
    for j in range(100): #go by columns (by X)
        for n in range(100):
            P[i, j,n] = testfn3b(np.array([[X[i, j,n], Y[i, j,n], Z[i,j,n]]]))[0]


num_attempts = 300  
history_x, history_y = [], []  
local_minimum_x, local_minimum_y, local_minimum_z, local_minimum_p= [], [],[],[]

best_x=None
best_y=None
best_z=None
best_p =float('inf')
for _ in range(num_attempts):
    stagnation_count = 0
    x = random.randrange(x_min + d, x_max - d)
    y = random.randrange(y_min + d, y_max - d)
    z=random.randrange(z_min + d, z_max - d)
    
    temp_x = [x]  
    temp_y = [y]
    temp_z = [z]
    try_d = d

    while try_d > 1:
        current_val = testfn3b(np.array([[x, y,z]]))[0]

        
        neighbors = [
            (x + try_d, y,z), (x - try_d, y,z),
            (x, y + try_d,z), (x, y - try_d,z),
            (x, y ,z+ try_d), (x, y ,z- try_d),
        ]
        
       
        best_local_x=x
        best_local_y= y
        best_local_z=z
        best_value  = current_val

        for ix, iy,iz in neighbors:
            neighbor_value = testfn3b(np.array([[ix, iy, iz]]))[0]
            if neighbor_value < best_value :
                best_local_x=ix
                best_local_y=iy
                best_local_z=iz
                best_value =neighbor_value

        if best_value >= current_val:
            try_d -= d // 2
        else:
            x, y,z = best_local_x, best_local_y,best_local_z
            temp_x.append(x)
            temp_y.append(y)
            temp_z.append(z)

    final_p = testfn3b(np.array([[temp_x[-1], temp_y[-1], temp_z[-1]]]))[0]

    if (temp_x[-1], temp_y[-1], temp_z[-1]) not in zip(local_minimum_x, local_minimum_y, local_minimum_z):
        local_minimum_x.append(temp_x[-1])
        local_minimum_y.append(temp_y[-1])
        local_minimum_z.append(temp_z[-1])
        local_minimum_p.append(testfn3b(np.array([[x, y,z]]))[0])
        if final_p < best_p:
            best_x, best_y, best_z, best_p = temp_x[-1], temp_y[-1], temp_z[-1], final_p
    history_x.append(temp_x)
    history_y.append(temp_y)
    history_y.append(temp_z)


print(best_p)