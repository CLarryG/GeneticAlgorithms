import random
import numpy as np
import matplotlib.pyplot as plt


# --- Fitness Function ---
def fitness(chromosome, weights, n_bins):
    """Compute fitness = 100 / (1 + d), where d = difference between heaviest and lightest bins."""
    bin_sums = [0] * n_bins
    for i, bin_idx in enumerate(chromosome):
        bin_sums[bin_idx] += weights[i]
    d = max(bin_sums) - min(bin_sums)
    fit = 100 / (1 + d)
    return fit, d


# --- Initialize Random Population ---
def initialize_population(n_items, n_bins, pop_size):
    """Generate initial population of random bin assignments."""
    return [np.random.randint(0, n_bins, n_items).tolist() for _ in range(pop_size)]


# --- Tournament Selection ---
def tournament_selection(population, fitnesses, k=3):
    """Select a parent using tournament selection."""
    selected = random.sample(range(len(population)), k)
    best = max(selected, key=lambda i: fitnesses[i])
    return population[best]


# --- Uniform Crossover ---
def uniform_crossover(parent1, parent2, pc):
    """Uniform crossover: each gene has 50% chance of coming from either parent."""
    if random.random() > pc:
        return parent1.copy(), parent2.copy()  # no crossover
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    return child1, child2


# --- Random Reassignment Mutation ---
def mutate(chromosome, n_bins, mutation_rate=0.05):
    """Randomly reassign some items to new bins."""
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, n_bins - 1)
    return chromosome


# --- Genetic Algorithm Core (with statistics and evaluation limit) ---
def genetic_algorithm(weights, n_bins, pop_size=50, pc=0.8, mutation_rate=0.05,
                      elitism=True, max_evaluations=10000, seed=None):
    """Run GA until 10,000 fitness evaluations have been reached. Track mean & std."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_items = len(weights)
    population = initialize_population(n_items, n_bins, pop_size)

    best_chromosome = None
    best_diff = float('inf')
    best_fitness = 0
    fitness_evals = 0
    generation = 0

    # Statistics tracking
    best_history = []
    mean_history = []
    std_history = []

    # Evaluate initial population
    evals = [fitness(ch, weights, n_bins) for ch in population]
    fitness_evals += len(evals)

    while fitness_evals < max_evaluations:
        fitnesses = [f[0] for f in evals]
        diffs = [f[1] for f in evals]

        # Compute statistics
        gen_best_fit = max(fitnesses)
        gen_mean_fit = np.mean(fitnesses)
        gen_std_fit = np.std(fitnesses)

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_diff = diffs[gen_best_idx]
        if gen_best_diff < best_diff:
            best_diff = gen_best_diff
            best_fitness = gen_best_fit
            best_chromosome = population[gen_best_idx]

        # Record stats
        best_history.append(gen_best_fit)
        mean_history.append(gen_mean_fit)
        std_history.append(gen_std_fit)

        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Gen {generation:4d} | Evals: {fitness_evals:5d} | "
                  f"Best diff={best_diff:.2f} | Best fit={gen_best_fit:.4f} | "
                  f"Mean fit={gen_mean_fit:.4f} | Std={gen_std_fit:.4f}")

        # --- New Population (Generational Replacement + Elitism) ---
        new_population = []
        if elitism:
            new_population.append(best_chromosome.copy())

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = uniform_crossover(parent1, parent2, pc)
            new_population.append(mutate(child1, n_bins, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, n_bins, mutation_rate))

        population = new_population

        # Evaluate new generation
        evals = [fitness(ch, weights, n_bins) for ch in population]
        fitness_evals += len(evals)
        generation += 1

    # Compute final bin weights for best chromosome
    final_bins = [0] * n_bins
    for i, b in enumerate(best_chromosome):
        final_bins[b] += weights[i]

    return (best_chromosome, final_bins, best_diff, best_fitness,
            best_history, mean_history, std_history, fitness_evals)


# --- Problem Definition ---
def generate_problem(problem_num):
    """Generates weights for Problem 1 or Problem 2."""
    if problem_num == 1:
        n_items = 500
        n_bins = 10
        weights = [i for i in range(1, n_items + 1)]
    elif problem_num == 2:
        n_items = 500
        n_bins = 50
        weights = [(i ** 2) / 2 for i in range(1, n_items + 1)]
    else:
        raise ValueError("Invalid problem number. Use 1 or 2.")
    return weights, n_bins


# --- Run a Single Trial ---
if __name__ == "__main__":
    problem_number = 1  # change to 2 for second problem
    weights, n_bins = generate_problem(problem_number)

    seed = random.randint(0, 100000)
    print(f"\nRunning GA Trial for Problem {problem_number} with seed = {seed}...\n")

    (best_chrom, bin_totals, best_diff, best_fit,
     best_hist, mean_hist, std_hist, evals) = genetic_algorithm(
        weights, n_bins,
        pop_size=80,
        pc=0.8,
        mutation_rate=0.05,
        elitism=True,
        max_evaluations=10000,
        seed=seed
    )

    print("\n=== Final Results ===")
    print(f"Problem {problem_number}:")
    print(f"Total fitness evaluations: {evals}")
    print(f"Best difference (max-min): {best_diff:.4f}")
    print(f"Best fitness: {best_fit:.4f}")
    print(f"Bin totals: {bin_totals}")

    # --- Plot Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_hist, label="Best Fitness")
    plt.plot(mean_hist, label="Mean Fitness")
    plt.fill_between(range(len(mean_hist)),
                     np.array(mean_hist) - np.array(std_hist),
                     np.array(mean_hist) + np.array(std_hist),
                     color='orange', alpha=0.3, label="±1 Std Dev")
    plt.title(f"Fitness Convergence (Problem {problem_number})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

#https://chatgpt.com/share/69131b39-c708-8004-a42c-f81a1808fcff
#[1] OpenAI, “Explanation of Genetic Algorithm code and parameters for the Bin-Packing Problem,” ChatGPT, Nov. 11, 2025. [Online]. Available: https://chat.openai.com/