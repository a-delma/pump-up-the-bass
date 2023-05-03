import torch
from torch_geometric.nn import VGAE, GCNConv

def train(model, train_data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


# @torch.no_grad()
# def test(model, data):
#     model.eval()
#     z = model.encode(data.x, data.edge_index)
#     return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = 'cpu'
# from dataset import erdos_dataset
# dataset = erdos_dataset(1e-4)
# train_data, val_data, test_data = dataset[0]

def get_gae(train_data, out_dims, epochs=25, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if test_data is None:
    #     test_data = train_data

    # train_data, val_data, test_data = dataset[0]

    in_channels, out_channels = train_data.num_node_features, out_dims
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(1, epochs + 1):
        loss = train(model, train_data, optimizer)
        if epoch % 5 == 0:
            # auc, ap = test(model, test_data)
            print(f'Epoch: {epoch:03d}, Loss: {loss} ')
    return model

if __name__ == "__main__":
    from dataset import erdos_dataset
    dataset = erdos_dataset(1e-4)
    train_data = dataset[0]
    get_gae(train_data, 25, 50)