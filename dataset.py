# import torch_geometric.datasets.gnn_benchmark_dataset as DS
import torch_geometric.datasets.wikics as DS
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import torch_geometric.transforms as T
import torch
import networkx as nx
import math
import scipy.stats as stats
# import matplotlib.pyplot as plt


def plot_degree_dist(G):
    return
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=30)
    plt.xlim([0, 500])
    plt.show()

def main():
    t = erdos_dataset(1e-3)
    # t2 = powerlaw_dataset(1e-3)
    data= t[0]
    print(data.x.size(), data.y.size())

    # print(val.x.size(), val.y.size())
    print(data.train_mask)
    print(data.y[data.train_mask].size())
    # print(t[1])
    # print(t2[0])
    # print(t2[1])


def random_split(data, G):
    # data = T.RandomNodeSplit(num_val=0.2, num_test=0.1)(data)
    G.y = data.y
    G.train_mask = data.train_mask[:, 0]
    G.val_mask = data.val_mask[:, 0]
    G.test_mask = data.test_mask
    return G

def erdos_dataset(p=0):

    def erdos_transform(data: Data) -> Data:
        # print(data.train_mask.size(), data.test_mask.size())
        G : nx.DiGraph = to_networkx(data, node_attrs=['x'])
        original_edges = len(G.edges())
        G_random = nx.generators.random_graphs.fast_gnp_random_graph(len(G), p)
        G.add_edges_from(G_random.edges())
        added_edges = len(G.edges()) - original_edges

        G : Data = from_networkx(G, ['x'])
        # G = random_split(data, G)
        G.edges_added = added_edges
        G.y = data.y
        return G
    
    transform = T.Compose([
        erdos_transform,
        T.ToDevice('cuda'),
        T.RandomNodeSplit(num_val=0.2, num_test=0.1),
    ])

    train_data = DS.WikiCS('./data/', transform=transform, is_undirected=False)

    return train_data      

def powerlaw_dataset(p=0):

    def powerlaw_transform(data: Data) -> Data:
        G = to_networkx(data, node_attrs=['x'])
        f = math.factorial
        n = len(G)
        num_edges = int(f(n) // f(2) // f(n-2) * p)
        sum_degrees = sum(list(map(lambda x: x[1], G.degree())))
        ps = [G.degree(x) / sum_degrees for x in G]
        custm = stats.rv_discrete(name='custm', values=(range(0,n), ps))
        edges = [(src, tar) for src, tar in zip(custm.rvs(size=num_edges), custm.rvs(size=num_edges)) if src != tar]

        original_edges = len(G.edges())
        G.add_edges_from(edges)
        added_edges = len(G.edges()) - original_edges

        G = from_networkx(G)
        G.y = data.y
        G.edges_added = added_edges
        return G

    transform = T.Compose([
        powerlaw_transform,
        T.ToDevice('cuda'),
        T.RandomNodeSplit(num_val=0.2, num_test=0.1),
    ])

    train_data = DS.WikiCS('./data/', transform=transform, is_undirected=False)

    return train_data   

if __name__ == '__main__':
    main()