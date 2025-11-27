import numpy as np
import cv2
import random

img = cv2.imread("image.png", 0)
img = cv2.resize(img, (256,256))
h, w = img.shape

pop_size = 20
cells = [np.random.randint(0,256,(h,w)) for _ in range(pop_size)]

def fitness(mask):
    gx = cv2.Sobel(mask, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(mask, cv2.CV_64F, 0, 1)
    return np.mean(np.abs(gx) + np.abs(gy))

def mutate(x):
    m = x.copy()
    r = random.randint(1,3)
    for _ in range(r):
        i = random.randint(0,h-1)
        j = random.randint(0,w-1)
        m[i,j] = random.randint(0,255)
    return m

def crossover(a,b):
    mask = np.random.randint(0,2,(h,w))
    return a*mask + b*(1-mask)

iters = 15
for _ in range(iters):
    fits = [fitness(c) for c in cells]
    best = cells[np.argmax(fits)]
    new = []
    for _ in range(pop_size):
        p1, p2 = random.sample(cells, 2)
        c = crossover(p1,p2)
        if random.random() < 0.3:
            c = mutate(c)
        new.append(c)
    cells = new

edge_mask = cells[np.argmax([fitness(c) for c in cells])]
edges = cv2.Canny(edge_mask.astype(np.uint8), 60, 120)

cv2.imwrite("edges.png", edges)
