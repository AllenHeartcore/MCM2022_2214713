import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class GNNStack(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, emb=False):
        super(GNNStack, self).__init__()
        conv_model = GAT
        self.convs = nn.ModuleList([conv_model(input_dim, hidden_dim)] + 
                                   [conv_model(heads * hidden_dim, hidden_dim)] * (num_layers - 1))
        self.post_mp = nn.Sequential(nn.Linear(heads * hidden_dim, hidden_dim), 
                                     nn.Dropout(dropout), 
                                     nn.Linear(hidden_dim, output_dim))
        self.emb = emb

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=dropout, training=self.training)
        x = self.post_mp(x)
        if not self.emb: x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, **kwargs): 
        super(GAT, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_l = nn.Linear(in_channels, heads * out_channels)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(torch.Tensor(heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(heads, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index):
        H, C = heads, self.out_channels
        x_l, x_r = self.lin_l(x).reshape(-1, H, C), self.lin_r(x).reshape(-1, H, C)
        alpha_l, alpha_r = self.att_l * x_l, self.att_r * x_r
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r))
        out = out.reshape(-1, H * C)
        return out

    def message(self, x_j, alpha_i, alpha_j, index):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, negative_slope=negative_slope)
        alpha = softmax(alpha, index=index)
        alpha = F.dropout(alpha, p=dropout)
        out = x_j * alpha
        return out

    def aggregate(self, inputs, index, dim_size = None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')

def train(dataset):
    test_loader = loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = GNNStack(dataset.num_node_features, hidden_dim, dataset.num_classes)
    filter_fn = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            pred = model(batch)
            pred = pred[batch.train_mask]
            label = batch.y[batch.train_mask]
            loss = model.loss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        test_acc = test(test_loader, model)
        print("Epoch: %02d, Loss: %6.4f, TestAcc: %4.1f%%" % (epoch + 1, total_loss, test_acc * 100))

def test(loader, test_model):
    test_model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = test_model(data).max(dim=1)[1]
            label = data.y
        mask = data.test_mask
        pred = pred[mask]
        label = label[mask]
        correct += pred.eq(label).sum().item()
    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()
    return correct / total

num_layers     = 2
batch_size     = 32
hidden_dim     = 32
heads          = 2
epochs         = 100
lr             = 0.01
weight_decay   = 0.005
dropout        = 0.5
negative_slope = 0.2

dataset = Planetoid(root='dataset', name='Cora')
train(dataset)
