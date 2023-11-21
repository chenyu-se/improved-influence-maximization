from diffusion_dynamic import IndependentCascade
import numpy as np
import random
import operator
import heapq

population_size = 100
mutation_rate = 0.3
crossover_rate = 1.0
elite_number = 2
mc = 1

def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i in range(len(population)):
        current += fitnesses[i]
        if current > pick:
            return population[i]

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[start:end] = parent1[start:end]
    for gene in parent2:
        if gene not in child:
            for i in range(size):
                if child[i] is None:
                    child[i] = gene
                    break
    return child

def mutation(child, V):
    m = random.sample(V.difference(child), len(child))
    for i in range(len(child)):
        r = random.random()
        if r < mutation_rate:
            child[i] = m[i]
    return child
            
def fitness(child, diffuse, t0, duration):
    f = diffuse.diffuse_mc(child, mc=mc, t0=t0, duration=duration)
    return f

def find_max_indices(arr, k):
    max_values = heapq.nlargest(k, arr)
    indices = [i for i, num in enumerate(arr) if num in max_values]
    return indices
    

def genetic_algorithm(graph, diffuse, k, t0, duration):
    V = set(graph.nodes())
    population = [random.sample(V, k) for _ in range(population_size)]
    generations = population_size * k
    print("evolution start, total generations: " + str(generations))
    for gen in range(generations):
        print("generation: " + str(gen))
        #fitness
        fitnesses = [fitness(child, diffuse, t0, duration) for child in population]
        
        #select
        selected = [roulette_wheel_selection(population, fitnesses) for _ in range(population_size)]
        
        #crossover
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            if (random.random() < crossover_rate):
                child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)
            else:
                child1, child2 = parent1, parent2
            offspring += [child1, child2]
        
        #mutation
        mutated_offspring = [mutation(child, V) for child in offspring]
        
        #elitism
        indices = find_max_indices(fitnesses, elite_number)
        for i in indices:
            mutated_offspring[i] = population[i]
        #update population
        population = mutated_offspring
    
    #best seed
    fitnesses = [fitness(child, diffuse, t0, duration) for child in population]
    indices = find_max_indices(fitnesses, 1)
    return population[indices[0]], fitnesses[indices[0]]
        
        
        
        
        
        