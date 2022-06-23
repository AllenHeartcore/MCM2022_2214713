import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import ToSparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim,  hidden_dim)]
                                        +[GCNConv(hidden_dim, hidden_dim)] * (num_layers - 2)
                                        +[GCNConv(hidden_dim, output_dim)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)] * (num_layers - 1))
        self.softmax = nn.LogSoftmax()
        self.dropout = dropout
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    def forward(self, x, adj_t):
        out = x
        for conv, bn in zip(self.convs[:-1], self.bns):
            out = conv(out, adj_t)
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout)
        out = self.convs[-1](out, adj_t)
        out = self.softmax(out)
        return out

def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    out = model(data.x, data.adj_t)[train_idx]
    loss = loss_fn(out, data.y[train_idx].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({'y_true': data.y[split_idx['train']], 'y_pred': y_pred[split_idx['train']]})['acc']
    valid_acc = evaluator.eval({'y_true': data.y[split_idx['valid']], 'y_pred': y_pred[split_idx['valid']]})['acc']
    test_acc  = evaluator.eval({'y_true': data.y[split_idx['test']],  'y_pred': y_pred[split_idx['test']]})['acc']
    return train_acc, valid_acc, test_acc

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = data.to(device)
split_idx = dataset.get_idx_split()

hidden_dim = 256
num_layers = 3
lr         = 0.01
dropout    = 0.5
epochs     = 100

model = GCN(data.num_features, hidden_dim, dataset.num_classes, num_layers, dropout).to(device)
model.reset_parameters()
loss_fn = F.nll_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
evaluator = Evaluator(name='ogbn-arxiv')

for epoch in range(epochs):
    loss = train(model, data, split_idx['train'], optimizer, loss_fn)
    train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
    print('Epoch: %02d, Loss: %.4f, Train: %.2f%%, Valid: %.2f%%, Test: %.2f%%' \
          % (epoch + 1, loss, train_acc * 100, valid_acc * 100, test_acc * 100))
