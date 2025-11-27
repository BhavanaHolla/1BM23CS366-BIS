import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

start = np.array([0.1, 0.1])
goal = np.array([0.9, 0.9])
obstacles = [
    (np.array([0.5, 0.5]), 0.15),
    (np.array([0.3, 0.7]), 0.1),
    (np.array([0.7, 0.3]), 0.1)
]

n_waypoints = 5
dim = n_waypoints * 2
wolves = 30
iters = 200
xmin, xmax = 0, 1

def collision(p1, p2, c, r):
    d = p2 - p1
    f = p1 - c
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    e = np.dot(f, f) - r*r
    disc = b*b - 4*a*e
    if disc < 0: return False
    disc = np.sqrt(disc)
    t1 = (-b - disc) / (2*a)
    t2 = (-b + disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def fitness(x):
    w = x.reshape(-1, 2)
    p = np.vstack([start, w, goal])
    if np.any(p < xmin) or np.any(p > xmax): return 1e6
    pen = 0
    for i in range(len(p)-1):
        for c, r in obstacles:
            if collision(p[i], p[i+1], c, r):
                pen += 1e4
    return np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1)) + pen

pos = np.random.uniform(xmin, xmax, (wolves, dim))
alpha = beta = delta = None
fa = fb = fd = 1e9

for t in range(iters):
    for i in range(wolves):
        f = fitness(pos[i])
        if f < fa:
            fa, fb, fd = f, fa, fb
            alpha, beta, delta = pos[i], alpha, beta
        elif f < fb:
            fb, fd = f, fb
            beta, delta = pos[i], beta
        elif f < fd:
            fd = f
            delta = pos[i]
    a = 2 - 2 * t / iters
    for i in range(wolves):
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D1 = abs(C1 * alpha - pos[i])
        X1 = alpha - A1 * D1

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D2 = abs(C2 * beta - pos[i])
        X2 = beta - A2 * D2

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D3 = abs(C3 * delta - pos[i])
        X3 = delta - A3 * D3

        pos[i] = (X1 + X2 + X3) / 3
        pos[i] = np.clip(pos[i], xmin, xmax)

best = alpha.reshape(-1, 2)
path = np.vstack([start, best, goal])

fig, ax = plt.subplots()
ax.plot(path[:,0], path[:,1], marker="o")
ax.scatter(start[0], start[1])
ax.scatter(goal[0], goal[1])
for c, r in obstacles:
    circ = plt.Circle(c, r, alpha=0.3)
    ax.add_patch(circ)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect("equal")
plt.show()
