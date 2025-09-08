from genetic_all import *

import matplotlib.pyplot as plt
import numpy as np

num_variables = 100
x_min = -800
x_max = 800
pop_size = 400
num_of_generations = 5000

space = uniform_space(num_variables, x_min, x_max)

amp = space[1, :] / 100  




for i in range(5):

    best_fitness_history = []


    population = genrpop(pop_size, space)


    last_improvement_gen = 0  

    for gen in range(num_of_generations):

 
        fitness = testfn3b(population)

        # saving the best value
        best_fitness_history.append(np.min(fitness))
        
        

        selected_best_population, _ = selbest(population, fitness, [10,15,25 ])
        selected_random_population, _ = seltourn(population, fitness, 170)
        selected_random3_population,_=selrand(population,fitness,180)
        
        crossov(selected_random_population, 1, 0)
        

        
        mutx(selected_random_population,0.05 ,space)
        mutx(selected_random3_population,0.025 ,space)
        muta(selected_random_population, 0.025, amp, space)

       
        population = np.vstack((selected_best_population, selected_random_population, selected_random3_population))


    best_solution = population[np.argmin(testfn3b(population))]
    best_value = np.min(testfn3b(population))
    
    print(best_value)
    plt.plot(best_fitness_history)

print(best_value)

plt.show()
