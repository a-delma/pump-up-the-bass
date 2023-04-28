import numpy as np
import torch
from numpy import linalg as LA
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx


class SpectralEmbedding:
        def __init__(self, d: Data):
                self.G = to_networkx(d)
                self.d = d
                self.A = nx.adjacency_matrix(self.G).todense()
                self.dsd = compute_dsd_embedding(self.A)
        def get_embeddings():
                node_emb = torch.tensor(self.dsd)
                # concatenate node embeddings with node features
                node_emb = torch.cat((node_emb, self.d.x), 1)
                return node_emb
                
  

def compute_dsd_embedding(A, t = -1, gamma = 1, is_normalized = True):
    """
    Code for computing the cDSD and normalized cDSD embedding
    NOTE: This does not return the distance matrix. Additional pairwise distance computations
          need to be done in order to produce the distance matrix after the embedding is obtained
          
    parameters:
        A: The square adjacency matrix representation of a network
        t: The number of random walks for the DSE computation. Setting it to `-1` sets this to infinity
        is_normalized: If set to true, normalizes the embedding at the end
        
    output:
        A n x n output embedding. NOTE: since the dimension of X is n x n, when n is large, it is more space-efficient to use the reduced dimension representation of
        x, using the function `compute_dsd_reduced_embedding` function for `gamma` == 1
    """
    n, _ = A.shape
    e    = np.ones((n, 1))
    
    # compute the degree vector and the Transition matrix `P`
    d = A @ e
    P = A/d

    # Compute scaling `W`, which is a rank one matrix used for removing the component from P with eigenvalue 1
    W     = (1 / np.sum(d)) * np.dot(e, (e * d).T)

    # This resulting `P'` has all eigenvalues less than 1
    P1 = gamma * (P - W)
    
    # X = (I - P')^{-1}, which is the cDSD embedding
    X  = np.linalg.pinv(np.eye(n, n) - P1)

    if t > 0:
        """
        This is computationally inefficient for larger t
        """
        P1_t = np.eye(n, n) - np.linalg.matrix_power(P1, t)
        # (I - P')^{-1}(I - (P')^t) = I + P' + P'^2 + ... + P'^{t-1}
        X    = np.matmul(X, P1_t)
    
    if is_normalized == False:

        return X

        
if __name__ == "__main__":
    from dataset import erdos_dataset
    ed,_,_ = erdos_dataset()
    spec_emb = SpectralEmbedding(ed[0]).get_embeddings()