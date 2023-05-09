import numpy as np
import torch
from numpy import linalg as LA
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SpectralEmbedding:
    def __init__(self, d: Data):
        self.G = to_networkx(d)
        self.d = d
        self.A = nx.adjacency_matrix(self.G).todense()
        
    def get_embeddings(self, fromfile = False):
        if fromfile:
            try:
                self.dsd = torch.load('dse_600.pt')
            except:
                self.dsd = compute_dsd_reduced_embedding(self.A, dims=self.d.num_node_features)
                torch.save(self.dsd, 'dse_600.pt')
        else:
            self.dsd = compute_dsd_reduced_embedding(self.A, dims=self.d.num_node_features)
            
        node_emb : torch.Tensor = torch.tensor(self.dsd)
        # node_emb.to("cuda")
        # concatenate node embeddings with node features
        node_emb = torch.cat([node_emb.to('cuda'), self.d.x.to('cuda')], -1)
        return node_emb.type(torch.FloatTensor)

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
    A = np.array(A)
    indeces = np.diag_indices(n)[0]
    indeces = indeces[A.diagonal() == 0]
    A[indeces, indeces] = 1
    
    # compute the degree vector and the Transition matrix `P`
    d = A @ e
    P = A/d
    t_ = np.multiply(e, d).T 
    # Compute scaling `W`, which is a rank one matrix used for removing the component from P with eigenvalue 1
    W     = (1 / np.sum(d)) * np.dot(e, t_)

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


def compute_dsd_reduced_embedding(A, dims = 50):
    """
    Performs the dimension reduction on the normalized DSD embedding, returning a
    reduced dimension matrix.
    
    parameters:
        A: A n x n numpy matrix representing the adjacency matrix 
        dims(d): The reduced dimension 
        
    output:
        A n x d dimensional embedding of the network.
    """
    
    n, _  = A.shape
    A = np.array(A)
    indeces = np.diag_indices(n)[0]
    indeces = indeces[A.diagonal() == 0]
    A[indeces, indeces] = 1
    # Get the normalized adjacency matrix N = D^{-1/2} A D^{-1/2}, L = I - N
    d1_2 = np.sqrt(A @ np.ones((n, 1)))
    N    = (A / d1_2) / d1_2.T       
    L    = np.eye(n, n) - N
    
    # Get the eigendecomposition of the laplacian
    lvals, X_ls  = LA.eig(L + np.eye(n, n))
    
    # Get the smallest `dims` eigencomponents
    l_ids = np.argsort(lvals)[1 : dims + 1]
    l_r   = lvals[l_ids] - 1
    
    X_lr  = X_ls[:, l_ids] / d1_2
    
    return np.real(X_lr) * np.real(l_r).reshape(1, -1)
        
if __name__ == "__main__":
    from dataset import erdos_dataset
    ed = erdos_dataset()
    spec = SpectralEmbedding(ed[0])
    spec_emb = spec.get_embeddings()
    print("hello")
    print(spec_emb.size())