import numpy as np
import random

np.random.seed(0)

n = 8
matrix = np.random.randint(-10, 10, (n, n))

def fitness(x):
    s = 0
    for i in range(n):
        s += matrix[x[i], x[(i+1)%n]]
        s += matrix[x[(i+1)%n], x[i]]
    return -s

def levy():
    beta = 1.5
    sigma = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) /
            (np.math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, n)
    v = np.random.normal(0, 1, n)
    step = u / abs(v)**(1/beta)
    return step

nests = [np.random.permutation(n) for _ in range(10)]
fitnesses = [fitness(x) for x in nests]

for _ in range(2000):
    i = random.randint(0, 9)
    step = levy()
    new = nests[i].copy()
    a, b = random.sample(range(n), 2)
    new[a], new[b] = new[b], new[a]
    fnew = fitness(new)
    if fnew < fitnesses[i]:
        nests[i] = new
        fitnesses[i] = fnew
    j = random.randint(0, 9)
    if random.random() < 0.25:
        nests[j] = np.random.permutation(n)
        fitnesses[j] = fitness(nests[j])

best = nests[np.argmin(fitnesses)]
print("Best Seating:", best)
print("Happiness Score:", -fitness(best))
