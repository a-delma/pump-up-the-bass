# import torch_geometric.datasets.gnn_benchmark_dataset as DS
import torch_geometric.datasets.wikics as DS
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx
import math
import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=30)
    plt.xlim([0, 500])
    plt.show()

def main():
    t, _, _ = erdos_dataset(1e-2)
    t2, _, _ = powerlaw_dataset(1e-2)
    print(t[0])
    print(t2[0])
    # print(t2[1])

def erdos_dataset(p=0):

    def erdos_transform(data: Data) -> Data:
        G = to_networkx(data)
        plot_degree_dist(G)

        G_random = nx.generators.random_graphs.fast_gnp_random_graph(len(G), p)
        G.add_edges_from(G_random.edges())
        plot_degree_dist(G)

        return from_networkx(G)

    train_data = DS.WikiCS('./data/', transform=erdos_transform, is_undirected=False)
    # val_data = DS.GNNBenchmarkDataset('./data/', 'PATTERN', 'val', transform=erdos_transform)
    # test_data = DS.GNNBenchmarkDataset('./data/', 'PATTERN', 'test', transform=erdos_transform)

    return train_data, 1, 1        

def powerlaw_dataset(p=0):

    def erdos_transform(data: Data) -> Data:
        G = to_networkx(data)
        plot_degree_dist(G)
        f = math.factorial
        n = len(G)
        num_edges = int(f(n) // f(2) // f(n-2) * p)
        print(num_edges, n)
        sum_degrees = sum(list(map(lambda x: x[1], G.degree())))
        ps = [G.degree(x) / sum_degrees for x in G]
        custm = stats.rv_discrete(name='custm', values=(range(0,n), ps))
        edges = [(src, tar) for src, tar in zip(custm.rvs(size=num_edges), custm.rvs(size=num_edges)) if src != tar]
        G.add_edges_from(edges)
        plot_degree_dist(G)
        return from_networkx(G)

    train_data = DS.WikiCS('./data/', transform=erdos_transform, is_undirected=False)
    # val_data = DS.GNNBenchmarkDataset('./data/', 'TSP', 'val', transform=erdos_transform)
    # test_data = DS.GNNBenchmarkDataset('./data/', 'TSP', 'test', transform=erdos_transform)

    return train_data, 1, 1     

if __name__ == '__main__':
    main()