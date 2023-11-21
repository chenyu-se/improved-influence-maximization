from diffusion_dynamic import IndependentCascade

mc = 1

def greedy(graph, diffuse, k, t0, duration):
	S = set()
	A = set(graph.nodes)
	while len(S) < k:
		node_diffusion = {}
		for node in A:
			S.add(node)
			node_diffusion[node] = diffuse.diffuse_mc(S, mc=mc, t0=t0, duration=duration)
			S.remove(node)
		max_node = max(node_diffusion.items(), key=lambda x: x[1])[0]
		S.add(max_node)
		A.remove(max_node)
	return S
