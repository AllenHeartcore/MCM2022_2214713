import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.transforms import ToSparseTensor
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList([GCNConv(input_dim,  hidden_dim)]
                                        +[GCNConv(hidden_dim, hidden_dim)] * (num_layers - 2)
                                        +[GCNConv(hidden_dim, output_dim)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim)] * (num_layers - 1))
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dropout
        self.return_embeds = return_embeds
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
            out = torch.nn.functional.relu(out)
            out = torch.nn.functional.dropout(out, p=self.dropout)
        out = self.convs[-1](out, adj_t)
        if not self.return_embeds: out = self.softmax(out)
        return out

class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()
        self.node_encoder = AtomEncoder(hidden_dim)
        self.gcn_node = GCN(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.pool = global_mean_pool
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
    def reset_parameters(self):
        self.gcn_node.reset_parameters()
        self.linear.reset_parameters()
    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)
        embed = self.gcn_node(embed, edge_index)
        embed = self.pool(embed, batch)
        out = self.linear(embed)
        return out

def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        is_labeled = batch.y == batch.y
        out = model(batch)[is_labeled]
        loss = loss_fn(out, batch.y[is_labeled].type(torch.float32).reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True,  num_workers=0)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(dataset[split_idx["test"]],  batch_size=32, shuffle=False, num_workers=0)

hidden_dim = 256
num_layers = 5
lr         = 0.001
dropout    = 0.5
epochs     = 30

model = GCN_Graph(hidden_dim, dataset.num_tasks, num_layers, dropout).to(device)
model.reset_parameters()
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
evaluator = Evaluator(name='ogbg-molhiv')

for epoch in range(epochs):
    print('Training...')
    loss = train(model, device, train_loader, optimizer, loss_fn)
    print('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)
    valid_result = eval(model, device, valid_loader, evaluator)
    test_result = eval(model, device, test_loader, evaluator)
    train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], valid_result[dataset.eval_metric], test_result[dataset.eval_metric]
    print('Epoch: %02d, Loss: %.4f, Train: %.2f%%, Valid: %.2f%%, Test: %.2f%%' \
          % (epoch + 1, loss, train_acc * 100, valid_acc * 100, test_acc * 100))
