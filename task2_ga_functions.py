from genetic_all import *

import matplotlib.pyplot as plt
import numpy as np

num_variables = 10
x_min = -800
x_max = 800
pop_size = 100
num_of_generations = 500

space = uniform_space(num_variables, x_min, x_max)


amp = space[1, :] / 100  


for i in range(5):

    best_fitness_history = []

    
    population = genrpop(pop_size, space)

    
    for gen in range(num_of_generations):

    
        fitness = testfn3b(population)

        
        best_fitness_history.append(np.min(fitness))
        
        # elitaristicky vyber
        # funkcia zoradi jedince podla uspesnoti a vytvori
        # novu populaciu podla n_listu
        # Elite selection (take the top 20 individuals)
        selected_best_population, _ = selbest(population, fitness, [20, ])
        # tournament selection (select 80 random individuals)
        # turnajovy vyber
        # funkcia vykona "zapasy" medzi nahodne vybranymi jedincami
        # lepsi jedinec bude skopirovany do novej pop
        selected_random_population, _ = seltourn(population, fitness, 80)

        # krizenie s vyberom poctu bodov krizenia
        # funkcia zkrizi pary jedincov tak, ze sa podla parametra pts
        # vytvoria body krizenia medzi ktorymi sa vymenia geny jedincov
        # crossbreeding (crossover) to create offspring.
        # vytvorenie listu indexov dvojic, zamiesanie ak mode==0
        crossov(selected_random_population, 1, 0)

        # obycajna mutacia
        # funckia zmeni nahodne vybrane geny na cisla
        # v rozsahu danom parametrom space
        # intenzita mutacie je v parametri rate
        mutx(selected_random_population,0.1,space)
        #muta(selected_random_population, 0.1, amp, space)

        
        population = np.vstack((selected_best_population, selected_random_population))


    best_solution = population[np.argmin(testfn3b(population))]
    best_value = np.min(testfn3b(population))

    print(f"Best: {best_solution}")
    print(f"Fitness: {best_value}\n")
    
    plt.plot(best_fitness_history)

plt.show()
