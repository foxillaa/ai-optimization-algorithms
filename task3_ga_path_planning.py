import numpy as np
import matplotlib.pyplot as plt
import math

from genetic_all import *




B = [ [0, 0], [25, 68], [12, 75],[32, 17],[51, 64], [20, 19], [52, 87], 
      [80, 37], [35, 82],[2, 15],[50, 90],[13, 50],[85, 52],[97, 27],
      [37, 67],[20, 82],[49, 0],[62, 14],[7, 60],[100, 100] ]


pop_size = 800
generations = 1000

elite_rate = 0.08
parent_rate = 0.82
new_rate = 1 - (elite_rate + parent_rate)

elite_size = int(elite_rate * pop_size)
parent_size = int(parent_rate * pop_size)
new_size = int(new_rate * pop_size)


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def decode(chromosome):
    
    middle_cities = [g + 2 for g in chromosome]
    
    route = [1] + middle_cities + [20]
    return route


def fitness(chromosome):
    route = decode(chromosome)
    total_length = 0

    for i in range(len(route) - 1):
        x1, y1 = B[route[i] - 1]
        x2, y2 = B[route[i+1] - 1]
        total_length += distance(x1, y1, x2, y2)

    return total_length


def crossover(parent1, parent2):
    size = len(parent1)
    c1, c2 = sorted(np.random.choice(range(size), 2, replace=False))

    child1 = np.array([-1] * size)
    child2 = np.array([-1] * size)

    child1[c1:c2] = parent1[c1:c2]
    child2[c1:c2] = parent2[c1:c2]

    fill_pos = c2
    for gene in parent2:
        if gene not in child1:
            if fill_pos == size:
                fill_pos = 0
            child1[fill_pos] = gene
            fill_pos += 1

    fill_pos = c2
    for gene in parent1:
        if gene not in child2:
            if fill_pos == size:
                fill_pos = 0
            child2[fill_pos] = gene
            fill_pos += 1

    check_unique(child1)
    check_unique(child2)

    return child1, child2


def check_unique(child):
    return len(set(child)) == len(child)

all_runs_fitness = []
best_overall_dist = float("inf")
best_overall_route = None

for run in range(10):

  
    population = genrpop_perm(pop_size, 18)

    best_fitness = []

    for gen in range(generations):

  
        fitness_znac = np.array([fitness(chrom) for chrom in population])

        population_copy1 = population.copy()
        population_copy2 = population.copy()

        # elite part
        selected_best_population, _ = selbest(population_copy1, fitness_znac, [elite_size, ], reverse=False)

        # parents
        selected_random_population, _ = seltourn(population_copy2, fitness_znac, parent_size, reverse=False)

        children = []
        
        for i in range(0, len(selected_random_population) - 1, 2):
            c1, c2 = crossover(selected_random_population[i], selected_random_population[i + 1])
            children.extend([c1, c2])

        selected_random_population = np.array(children)

        swapgen(selected_random_population, 0.05)
        

        
        new = genrpop_perm(new_size, 18)

       
        population = np.vstack((selected_best_population, selected_random_population, new))

    
        new_fit = [fitness(chrom) for chrom in population]
        best_fitness.append(min(new_fit))


    all_runs_fitness.append(best_fitness)


    final_fit = np.array([fitness(chrom) for chrom in population])
    best_idx = np.argmin(final_fit)
    best_dist = final_fit[best_idx]
    best_chrom = population[best_idx]
    best_route = decode(best_chrom)


    print("Best dist: ", best_dist)

    if best_dist < best_overall_dist:
        best_overall_dist = best_dist
        best_overall_route = best_route


plt.figure(figsize=(10, 5))
for i, fitness_data in enumerate(all_runs_fitness):
    plt.plot(fitness_data, label=f"Run {i+1}")

plt.legend()
plt.show()


full_x = []
full_y = []
for city in best_overall_route:
    cx, cy = B[city - 1]
    full_x.append(cx)
    full_y.append(cy)

plt.figure(figsize=(8, 6))
plt.plot(full_x, full_y, "-o", color="red")

for i, city_id in enumerate(best_overall_route):
    plt.annotate(str(city_id), (full_x[i], full_y[i]))

plt.show()
