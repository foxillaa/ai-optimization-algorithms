import numpy as np
import math
import random
import matplotlib.pyplot as plt

from genetic_all import *

def death_penalty(pop: np.ndarray) -> np.ndarray:

    n = pop.shape[0]
    fitness = np.zeros(n)
    for i in range(n):
        x1, x2, x3, x4, x5 = pop[i]
        J = 0.04 * x1 + 0.07 * x2 + 0.11 * x3 + 0.06 * x4 + 0.05 * x5

        p1 = x1 + x2 + x3 + x4 + x5
        p2 = x1 + x2
        p3 = -x4 + x5
        p4 = -0.5 * x1 - 0.5 * x2 + 0.5 * x3 + 0.5 * x4 - 0.5 * x5

        if p1 > 1e7 or p2 > 2.5e6 or p3 > 0 or p4 > 0:
            fitness[i] = float("-inf")
        else:
            fitness[i] = J

    return fitness


def step_penalty(pop: np.ndarray) -> np.ndarray:

    n = pop.shape[0]
    fitness = np.zeros(n)
    c = 1e6
    for i in range(n):
        x1, x2, x3, x4, x5 = pop[i]
        J = 0.04 * x1 + 0.07 * x2 + 0.11 * x3 + 0.06 * x4 + 0.05 * x5

        p1 = x1 + x2 + x3 + x4 + x5 - 1e7
        p2 = x1 + x2 - 2.5e6
        p3 = -x4 + x5
        p4 = -0.5 * x1 - 0.5 * x2 + 0.5 * x3 + 0.5 * x4 - 0.5 * x5

        fitness[i] = J
        p = sum([p1 > 0, p2 > 0, p3 > 0, p4 > 0])

        if p > 0:
            fitness[i] -= p * c

    return fitness


def proportional_penalty(pop: np.ndarray) -> np.ndarray:
    n = pop.shape[0]
    fitness = np.zeros(n)
    a=0
    b=1
    c=1

    for i in range(n):

        x1, x2, x3, x4, x5 = pop[i]
        J = 0.04 * x1 + 0.07 * x2 + 0.11 * x3 + 0.06 * x4 + 0.05 * x5

        p1 = x1 + x2 + x3 + x4 + x5 - 1e7
        p2 = x1 + x2 - 2.5e6
        p3 = -x4 + x5
        p4 = -0.5 * x1 - 0.5 * x2 + 0.5 * x3 + 0.5 * x4 - 0.5 * x5

        fitness[i] = J

        if p1 > 0:
            fitness[i] -= (a + c * (p1 ** b))
        if p2 > 0:
            fitness[i] -= (a + c * (p2 ** b))
        if p3 > 0:
            fitness[i] -= (a + c * (p3 ** b))
        if p4 > 0:
            fitness[i] -= (a + c * (p4 ** b))

    return fitness

pop_size = 700
generations = 2000
x_min = 0
x_max = 5e6

space = uniform_space(5, x_min, x_max)

#penalty_func = death_penalty
#penalty_func = proportional_penalty
penalty_func = step_penalty

best_fitness_history = []
best_overall_fitness = float("-inf")
best_overall_x = None

for run in range(5):


    population = np.random.randint(x_min, x_max, size=(pop_size, 5))
    best_fitness = []

    for gen in range(generations):
        fitness_vals = penalty_func(population)

        selected_best_population, _ = selbest(population, fitness_vals, [70, ], True)

        selected_random_population, _ = seltourn(population, fitness_vals, 630, True)


        crossov(selected_random_population, 1, 0)


        mutx(selected_random_population, 0.02, space)

        population = np.vstack((selected_best_population, selected_random_population))

        new_fitness_vals = penalty_func(population)

        best_fitness.append(np.max(new_fitness_vals))

    best_fitness_history.append(best_fitness)

    final_fitness = penalty_func(population)
    
    #np.argmax(final_fitness) nájde index maximálnej hodnoty vo final_fitness.
    idx_best = np.argmax(final_fitness)
    best_x = population[idx_best]
    best_val = final_fitness[idx_best]

    x1, x2, x3, x4, x5 = best_x
    i=0
    print(f"fitness {i+1}: {best_val:.2f}")
    print(f"solution {i+1}: x1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}, x4 = {x4:.2f}, x5 = {x5:.2f}")
    
    if best_val > best_overall_fitness:
        best_overall_fitness = best_val
        best_overall_x = best_x

x1, x2, x3, x4, x5 = best_overall_x

print(f"Best fitness: {best_overall_fitness:.2f}")
print(f"Best solution: x1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}, x4 = {x4:.2f}, x5 = {x5:.2f}")

plt.figure(figsize=(10, 5))
for i, fitness_data in enumerate(best_fitness_history):
    plt.plot(fitness_data, label=f"Run {i + 1}")

plt.legend()

plt.show()
