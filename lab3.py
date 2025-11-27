import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

start = np.array([0.1, 0.1])
goal = np.array([0.9, 0.9])
obstacles = [
    (np.array([0.5, 0.5]), 0.15),
    (np.array([0.3, 0.7]), 0.1),
    (np.array([0.7, 0.3]), 0.1),
]

n_waypoints = 5
dim = n_waypoints * 2
n_particles = 40
n_iters = 200
w = 0.7
c1 = 1.5
c2 = 1.5
xmin, xmax = 0.0, 1.0

def segment_circle_collision(p1, p2, center, radius):
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return False
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    return False

def path_length(points):
    return np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))

def fitness(x):
    waypoints = x.reshape(-1, 2)
    path = np.vstack([start, waypoints, goal])
    if np.any(path < xmin) or np.any(path > xmax):
        return 1e6 + np.sum(np.clip(path - xmax, 0, None) + np.clip(xmin - path, 0, None))
    penalty = 0.0
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        for c, r in obstacles:
            if segment_circle_collision(p1, p2, c, r):
                penalty += 1e4
    return path_length(path) + penalty

positions = np.random.uniform(xmin, xmax, (n_particles, dim))
velocities = np.zeros((n_particles, dim))
pbest_pos = positions.copy()
pbest_val = np.array([fitness(p) for p in positions])
gbest_idx = np.argmin(pbest_val)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]

for it in range(n_iters):
    r1 = np.random.rand(n_particles, dim)
    r2 = np.random.rand(n_particles, dim)
    velocities = (
        w * velocities
        + c1 * r1 * (pbest_pos - positions)
        + c2 * r2 * (gbest_pos - positions)
    )
    positions += velocities
    positions = np.clip(positions, xmin, xmax)
    for i in range(n_particles):
        val = fitness(positions[i])
        if val < pbest_val[i]:
            pbest_val[i] = val
            pbest_pos[i] = positions[i].copy()
            if val < gbest_val:
                gbest_val = val
                gbest_pos = positions[i].copy()

best_waypoints = gbest_pos.reshape(-1, 2)
best_path = np.vstack([start, best_waypoints, goal])

fig, ax = plt.subplots()
ax.plot(best_path[:, 0], best_path[:, 1], marker="o")
ax.scatter(start[0], start[1])
ax.scatter(goal[0], goal[1])
for c, r in obstacles:
    circle = plt.Circle(c, r, alpha=0.3)
    ax.add_patch(circle)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
plt.show()
