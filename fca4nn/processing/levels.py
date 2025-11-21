import numpy as np
import pickle
import json
from pprint import pprint

np.random.seed(0)

with open("CIFAR100/context_100_10.pkl", "rb") as pkl:
    context = pickle.load(pkl)
    lattice = context.lattice
    print(len(lattice))

# context = concepts.Context.fromstring('''
#            |human|knight|king |mysterious|
# King Arthur|  X  |  X   |  X  |          |
# Sir Robin  |  X  |  X   |     |          |
# holy grail |     |      |     |     X    |
# ''')
# lattice = context.lattice

# print(type(lattice[200]))
# print(lattice.join(lattice[4]))
# for e, i in lattice:
# print(e, i)

# print(lattice.infimum)
# print(lattice.infimum.upper_neighbors)
# print(lattice.infimum.upper_neighbors[0].upper_neighbors)
# print(lattice.infimum.upper_neighbors[1].upper_neighbors)
# print(lattice.infimum.upper_neighbors[0].upper_neighbors[0].upper_neighbors)
# print(lattice.infimum.upper_neighbors[0].upper_neighbors[0].upper_neighbors[0].upper_neighbors)


## BFS for computing intent levels
level = {}
infimum = lattice.infimum
e, i = infimum
level[str(i)] = 0

Q = [infimum]
while len(Q):
    v = Q.pop(0)
    ve, vi = v
    adjv = v.upper_neighbors
    for concept in adjv:
        Q.append(concept)
        e, i = concept
        if i not in level:
            level[str(i)] = level[str(vi)] + 1
        else:
            level[str(i)] = max(level[str(i)], level[str(vi)] + 1)
pprint(level, sort_dicts=False)
print(len(level))
with open("CIFAR100/intent_levels.json", "w") as json_file:
    json.dump(level, json_file, indent=4)
