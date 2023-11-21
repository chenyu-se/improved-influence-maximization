import numpy as np
import networkx as nx
from diffusion_dynamic import IndependentCascade
from greedy import greedy
import DGA

G = nx.DiGraph()
day = 86400
f = open("dataset/CollegeMsg.txt");
lines = [l.split() for l in f.readlines() if l.strip()]
min_t = None
max_t = None
for i in lines:
    if i[0].startswith("%"):
        continue
    G.add_edge(i[0], i[1], prob = 0.9)
    t = int(int(i[2]) / day)
    if min_t == None or t < min_t:
        min_t = t
    if max_t == None or t > max_t:
        max_t = t
edge_idx = {(u, v): i for i, (u, v) in enumerate(G.edges())}
temporal = []
for _ in range(max_t - min_t + 1):
    temporal.append([])
for i in lines:
    if i[0].startswith("%"):
        continue
    idx = edge_idx[(i[0], i[1])]
    t = int(int(i[2]) / day) - min_t
    temporal[t].append(idx)
sorted_temporal = [sorted(i) for i in temporal]
diffuse = IndependentCascade(G, edge_idx, temporal = temporal)
k = 5
t0 = 0
duration = 30
S, expand = DGA.genetic_algorithm(G, diffuse, k, t0, duration)
print(S)
print(expand)

#print(lines)
#for i in lines:
#pos = nx.random_layout(G) #make graph layout by FR
#nx.draw(G, pos, with_labels = True, alpha = 0.5)
#labels = nx.get_edge_attributes(G, 'weight')
#nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
#plt.show()
