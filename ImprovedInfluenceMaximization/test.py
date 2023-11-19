from diffusion import IndependentCascade
import networkx as nx
import heuristic
from greedy import greedy
from ivgreedy import ivgreedy
from celfpp import celfpp

G = nx.DiGraph()
dataset_name = "dolphins";
f = open("dataset/" + dataset_name + "/" + dataset_name + ".mtx")
lines = [l.split() for l in f.readlines() if l.strip()]
for i in lines:
    if i[0].startswith("%"):
        continue
    else:
        G.add_edge(i[0], i[1], prob = 0.1)
diffuse = IndependentCascade(G)
k = 10
S = greedy(G, diffuse, k)
expand = diffuse.diffuse_mc(S)
i_S = ivgreedy(G, k)
i_expand = diffuse.diffuse_mc(i_S)
e_S = heuristic.top_k(G, diffuse, k)
e_expand = diffuse.diffuse_mc(e_S)
f_S = celfpp(G, diffuse, k)
f_expand = diffuse.diffuse_mc(f_S)
print(S)
print(expand)
print(i_S)
print(i_expand)
print(e_S)
print(e_expand)
print(f_S)
print(f_expand)

#print(lines)
#for i in lines:
#pos = nx.random_layout(G) #make graph layout by FR
#nx.draw(G, pos, with_labels = True, alpha = 0.5)
#labels = nx.get_edge_attributes(G, 'weight')
#nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
#plt.show()
