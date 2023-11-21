import numpy as np
import networkx as nx
from tqdm.autonotebook import tqdm

class IndependentCascade(object):
    """
    Args:
        graph: networkx.DiGraph()
        edge_idx: {(u, v): i for i, (u, v) in enumerate(graph.edges())}
        temporal[t][i]: At time t, the edge (u, v) is effctive whose edge_idx == temporal[t][i]. 
        Given t, if there are many i to make temporal[t][i] the same, then the probability increases in power for edge (u, v).
        If temporal == None, then the network is static.
    """
    def __init__(self, graph, edge_idx, temporal = None):
        self.graph = graph
        self.sampled_graph = graph.copy()
        self.edge_idx = {(u, v): i for i, (u, v) in enumerate(self.graph.edges())}
        self.temporal = temporal
        self.reverse_edge_idx = {i: e for e, i in self.edge_idx.items()}
        self.prob_matrix = [self.graph.edges[self.reverse_edge_idx[i][0], self.reverse_edge_idx[i][1]]['prob'] for i in sorted(self.reverse_edge_idx.keys())]
    
    def sample_live_graph_mc(self, act_nodes, mc):
        edge_probs = {(u, v): d['prob'] for u, v, d in self.graph.edges().data()}
        probs = np.random.uniform(size=(mc, len(edge_probs)))
        self.sampled_graphs = []
        for p in probs:
            live_edges = np.array([p > self.prob_matrix]).astype(np.int8)
            self.sampled_graphs.append(live_edges)
    
    def diffuse_dynamic(self, act_nodes, t0, duration):
        edge_probs = {(u, v): d['prob'] for u, v, d in self.graph.edges().data()}
        new_act_nodes = set(act_nodes)
        live_edges = np.zeros([1, len(edge_probs)], dtype = np.int8)
        t1 = min(len(self.temporal), t0 + duration)
        for t in range(t0, t1):
            edge_idx_list = self.temporal[t]
            edge_count_map = {}
            for idx in edge_idx_list:
                if idx in edge_count_map:
                    edge_count_map[idx] += 1
                else:
                    edge_count_map[idx] = 1
            inc_nodes = set()
            for idx in edge_count_map.keys():
                edge = self.reverse_edge_idx[idx]
                count = edge_count_map[idx]
                if edge[0] not in new_act_nodes:
                    continue
                r = np.random.random()
                if r > edge_probs[edge] ** count:
                    live_edges[0][idx] = 1
                    inc_nodes.add(edge[1])
            new_act_nodes.update(inc_nodes)
        return len(new_act_nodes)
        
    def sample_live_graph(self, mcount):
        removed_edges_idx = np.where(self.sampled_graphs[mcount] == 0)[1].tolist()
        removed_edges = [self.reverse_edge_idx[i] for i in removed_edges_idx]
        Gp = self.graph.copy()
        Gp.remove_edges_from(removed_edges)
        self.sampled_graph = Gp

    def diffusion_iter(self, act_nodes):
        new_act_nodes = set(act_nodes)
        for node in act_nodes:
            for node2 in nx.algorithms.bfs_tree(self.sampled_graph, node).nodes():
                new_act_nodes.add(node2)
        for node in new_act_nodes:
            self.sampled_graph.nodes[node]['is_active'] = True

    def diffuse(self, act_nodes, mcount):
        self.sample_live_graph(mcount)
        nx.set_node_attributes(self.sampled_graph, False, name='is_active')

        for node in act_nodes:
            self.sampled_graph.nodes[node]['is_active'] = True
        
        self.diffusion_iter(act_nodes)
        active_nodes = [n for n, v in self.sampled_graph.nodes.data() if v['is_active']]
        self.graph.total_activated_nodes.append(len(active_nodes))

    def diffuse_mc(self, act_nodes, mc=10, t0=0, duration=30):
        if self.temporal == None:
            self.sample_live_graph_mc(act_nodes, mc)
            self.graph.total_activated_nodes = []
            for i in range(mc):
                self.diffuse(act_nodes, i)
            return sum(self.graph.total_activated_nodes) / float(mc)
        else:
            count_list = [self.diffuse_dynamic(act_nodes, t0, duration) for _ in range(mc)]
            return sum(count_list) / float(mc)

    def shapely_iter(self, act_nodes):
        nx.set_node_attributes(self.sampled_graph, False, name='is_active')

        for node in act_nodes:
            self.sampled_graph.nodes[node]['is_active'] = True

        self.diffusion_iter(act_nodes)
        active_nodes = [n for n, v in self.sampled_graph.nodes.data() if v['is_active']]
        return active_nodes

    def shapely_diffuse(self, nodes, mc=10, t0=0, duration = 30):
        self.sample_live_graph_mc(nodes, mc, t0, duration)
        for node in nodes:
            self.graph.nodes[node]['tmp'] = 0

        for c in tqdm(range(mc), desc='Shapely Monte Carlo', leave=False):
            self.sample_live_graph(c)
            active_nodes_with = []
            active_nodes_without = []
            for i in tqdm(range(len(nodes)), desc='Shapely Iter', leave=False):
                if i in active_nodes_with:
                    self.graph.nodes[node]['tmp'] = 0
                    continue
                active_nodes_with = self.shapely_iter(nodes[:i+1])
                active_nodes_without = self.shapely_iter(nodes[:i])
                self.graph.nodes[nodes[i]]['tmp'] +=  len(active_nodes_with) - len(active_nodes_without)

        for i in range(len(nodes)):
            self.graph.nodes[node]['tmp'] /= float(mc)

