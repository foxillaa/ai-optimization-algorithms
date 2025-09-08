from genetic_all import *

import matplotlib.pyplot as plt
import numpy as np

num_variables = 10
x_min = -512
x_max = 512
pop_size = 500
num_of_generations = 3000

space = uniform_space(num_variables, x_min, x_max)


amp = space[1, :] / 100  

for i in range(5):

    best_fitness_history = []

    population = genrpop(pop_size, space)

   
    last_improvement_gen = 0  


    for gen in range(num_of_generations):


        fitness = eggholder(population)

        
        best_fitness_history.append(np.min(fitness))
        
        
        
        selected_best_population, _ = selbest(population, fitness, [20,40,80, ])
        selected_random_population, _ = seltourn(population, fitness, 380)
        selected_random2_population,_=selrand(population, fitness, 360)
        new_pop=genrpop(120, space)

        crossov(selected_random_population, 4, 0)
        crossgrp(selected_random_population,2)
        
        mutx(selected_random2_population,0.5 ,space)
        muta(selected_random_population, 0.05, amp*2, space)
        mutx(selected_random_population,0.35 ,space)
        muta(selected_best_population,0.02,amp,space)
        

        population = np.vstack((selected_best_population, selected_random_population,selected_random2_population,new_pop))


    best_solution = population[np.argmin(eggholder(population))]
    best_value = np.min(eggholder(population))
    
    print(best_value)
    plt.plot(best_fitness_history)

#print(best_value)

plt.show()
