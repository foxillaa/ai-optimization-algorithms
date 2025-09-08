import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from genetic_all import *


x_min, x_max = -800, 800
y_min, y_max = -800, 800
d = 40 


xs = np.linspace(x_min, x_max, 100)
ys = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(xs, ys)


Z = np.zeros_like(X)
for i in range(100): # go by row (Y)
    for j in range(100): #go by columns (by X)
        Z[i, j] = testfn3b(np.array([[X[i, j], Y[i, j]]]))[0]

#finding local lows
num_attempts = 100  
history_x, history_y = [], []  
local_minimum_x, local_minimum_y, local_minimum_z = [], [],[]

best_x=None
best_y=None
best_z =float('inf')

for _ in range(num_attempts):
    
    x = random.randrange(x_min + d, x_max - d)
    y = random.randrange(y_min + d, y_max - d)
    
    temp_x = [x]  
    temp_y = [y]
    try_d = d


    while try_d > 1:
        current_val = testfn3b(np.array([[x, y]]))[0]

        
        neighbors = [
            (x + try_d, y), (x - try_d, y),
            (x, y + try_d), (x, y - try_d)
        ]
        
       
        best_local_x=x
        best_local_y= y
        best_value  = current_val

        for ix, iy in neighbors:
            neighbor_value = testfn3b(np.array([[ix, iy]]))[0]
            if neighbor_value < best_value :
                best_local_x=ix
                best_local_y=iy
                best_value =neighbor_value

        if best_value  >= current_val:
            try_d //= 2
        else:
            x, y = best_local_x, best_local_y
            temp_x.append(x)
            temp_y.append(y)


    final_z = testfn3b(np.array([[temp_x[-1], temp_y[-1]]]))[0]
    if (temp_x[-1], temp_y[-1]) not in list(zip(local_minimum_x, local_minimum_y)):
        local_minimum_x.append(temp_x[-1])
        local_minimum_y.append(temp_y[-1])
        local_minimum_z.append(final_z)

        if final_z < best_z:
            best_x, best_y, best_z = x, y, final_z

    history_x.append(temp_x)
    history_y.append(temp_y)


fig=plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.9)

ax.scatter(local_minimum_x, local_minimum_y, local_minimum_z, color='green', s=50, marker='o')

ax.scatter(best_x, best_y, best_z, color='red', s=100, marker='D')
print(best_x)
print(best_y)
print(best_z)

plt.show()
