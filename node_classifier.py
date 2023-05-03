
import torch.nn.functional as F
import torch_geometric.nn as nn
import torch
# from torchmetrics.classification import MulticlassAveragePrecision
from torcheval.metrics import MulticlassAUPRC
from gae import get_gae
from spectral import SpectralEmbedding

class simpleClassifer(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 300)
        self.fc2 = nn.Linear(300, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        # print(x.size())
        x = F.softmax(self.fc3(x), dim=-1)
        # print(x)
        return x

def train_model(model, data, y_true, epoch=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    y_true = y_true.to('cuda')
    loss_fn = torch.nn.CrossEntropyLoss()
    for e in range(epoch):
        optimizer.zero_grad()
        z = model(data)
        loss = loss_fn(z.squeeze(), y_true)
        loss.backward()
        optimizer.step()
        if e % 5 == 0:
            print(f'Epoch: {e} Loss: {loss}')

def test_model(model, data, y_true):
    model.eval()
    y_true = y_true.to('cuda')
    criteon = MulticlassAUPRC(num_classes=10)
    with torch.no_grad():
        z = model(data)
        criteon.update(z, y_true)
        auprc = criteon.compute()
        print(f"Precision: {auprc}, Size: {y_true.size()}")
    return auprc


if __name__ == "__main__":
    from dataset import erdos_dataset
    dataset = erdos_dataset(1e-4)
    train_data, val_data, test_data = dataset[0]
    embed_dim = 25
    gae = get_gae(train_data, test_data, embed_dim, 5)
    
    z = gae.encode(train_data.x, train_data.edge_index).detach()
    out_model = simpleClassifer(embed_dim).to('cuda')
    train_model(out_model, z, train_data.y, epoch=250)
    
    z_val = gae.encode(val_data.x, val_data.edge_index).detach()
    # prec = test_model(train_model, z_val, val_data.y)


    
    # test_model(val_data.x, val_data.y)