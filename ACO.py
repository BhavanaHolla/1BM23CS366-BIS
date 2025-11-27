import random
import numpy as np

jobs = [2,4,6,8,3,5,7,1]
machines = 3
alpha = 1
beta = 2
evap = 0.5
ants = 10
iters = 30

def makespan(sol):
    load=[0]*machines
    for j,m in enumerate(sol):
        load[m]+=jobs[j]
    return max(load)

def construct_solution(pher):
    sol=[]
    for j in range(len(jobs)):
        prob=[]
        for m in range(machines):
            prob.append((pher[j][m]**alpha)*((1/jobs[j])**beta))
        prob=np.array(prob)/sum(prob)
        m=np.random.choice(range(machines),p=prob)
        sol.append(m)
    return sol

def aco():
    pher=np.ones((len(jobs),machines))
    best=None
    bestfit=1e9
    for _ in range(iters):
        sols=[]
        fits=[]
        for _ in range(ants):
            s=construct_solution(pher)
            f=makespan(s)
            sols.append(s)
            fits.append(f)
            if f<bestfit: bestfit, best = f, s
        pher=(1-evap)*pher
        for s,f in zip(sols,fits):
            for j,m in enumerate(s):
                pher[j][m]+=1/f
    return best, bestfit

best, fit = aco()
print("Best assignment:", best)
print("Makespan:", fit)
