import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import edge_softmax, GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import pickle

MODEL_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_state.pth")

num_layers = 2

num_hidden = 256
activation = F.elu
feat_drop = 0
attn_drop = 0
negative_slope = 0.2
residual = True
lr = 0.005
weight_decay = 0
num_heads = 4
num_out_heads = 6

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


def main(args):
    # create the dataset
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]
    g = train_dataset.graph
    in_dim = train_dataset.features.shape[1]
    num_classes = train_dataset.labels.shape[1]
    heads = ([num_heads] * num_layers) + [num_out_heads]
    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))
    model = GAT(g, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual).to(device)
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay= weight_decay)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    return test(model, loss_fcn, device, test_dataloader)


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):
    loss_plot = []
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            print(features, features.shape, features.dtype)
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(features.float())
            # writer.add_graph(model, features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        writer.add_scalar('Training loss', loss_data, epoch)
        print("Epoch {:d} | Loss: {:.4f}".format(epoch, loss_data))

        if epoch % 5 == 0:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                features = torch.tensor(features).to(device)
                labels = torch.tensor(labels).to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
            scores_mean = np.array(scores).mean()
            print("F1-Score: {:.4f} ".format(scores_mean))
            loss_plot.append(scores_mean)
            writer.add_scalar('F1-score', scores_mean, epoch)
    with open("loss", "wb") as fp:
        pickle.dump(loss_plot, fp)
    plt.plot(np.arange(0,args.epochs,step = 5),loss_plot)
    plt.show()



def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.gat_layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def collate_fn(sample):
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if args.name is None:
        from datetime import datetime
        args.name = datetime.now().strftime("%Y-%m-%d_%H:%M")

    args.name = os.path.join("runs", args.name)
    os.makedirs(args.name, exist_ok=True)
    # writer = SummaryWriter(args.name)
    with SummaryWriter(args.name) as writer:
        main(args)
