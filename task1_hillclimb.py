from genetic_all import *

import math
import numpy as np
import random
import matplotlib.pyplot as plt

x_min =-800
x_max =800
d= 40 

x_graf=np.linspace(x_min,x_max, 500)  
y_graf=testfn3b(x_graf)
best_x = None
best_y = float('inf')

num_attempts = 20

history_x=[]
history_y=[]

local_minimum_x=[]
local_minimum_y=[]

for _ in range(num_attempts):
    x = random.randrange(x_min+d, x_max-d)
    temp_x=[x] 
    temp_y=[testfn3b(np.array([x]))]
    try_d=d

   
    while try_d>1:

        bod_x1 = max(x_min, x - try_d)
        bod_x2 = min(x_max, x + try_d)

        y=testfn3b(np.array([x]))
        y1=testfn3b(np.array([bod_x1]))
        y2=testfn3b(np.array([bod_x2]))

        if y1<y2 :
            mensia_hodnota_y=y1
            mensia_x=bod_x1

        else:
            mensia_hodnota_y=y2
            mensia_x=bod_x2


        if y <= mensia_hodnota_y:  
            try_d//=2
        else:
            x=mensia_x
            temp_x.append(x)
            temp_y.append(mensia_hodnota_y)

    

    if (temp_x[-1] not in local_minimum_x) and (temp_y[-1] not in local_minimum_y):
        local_minimum_x.append(temp_x[-1])
        local_minimum_y.append(temp_y[-1])
    
    history_x.append(temp_x)
    history_y.append(temp_y)

    


plt.figure(figsize=(10, 5))  
plt.plot(x_graf, y_graf, color="black")  


for i in range(len(history_x)):
    plt.scatter(history_x[i], history_y[i], color="blue", facecolors="none", s=15) 

for i in range(len(local_minimum_x)):
    plt.scatter(local_minimum_x[i], local_minimum_y[i], color="green",s=50)
    
for i in range(len(local_minimum_x)):
    plt.scatter(local_minimum_x[i], local_minimum_y[i], color="green",s=50)
    if(local_minimum_y[i]<best_y):
        best_y=local_minimum_y[i]
        best_x=local_minimum_x[i]
    
print(best_y)
plt.scatter([best_x], [best_y], color="red",marker="D", s=50)    

plt.show()