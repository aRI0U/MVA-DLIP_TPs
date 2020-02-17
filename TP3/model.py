import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

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

if __name__ == '__main__':
    import torch.nn.functional as F
    from torchviz import make_dot
    from dgl.data.ppi import LegacyPPIDataset
    from torch.utils.tensorboard import SummaryWriter

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

    train_dataset = LegacyPPIDataset(mode="train")
    in_dim = train_dataset.features.shape[1]
    num_classes = train_dataset.labels.shape[1]
    heads = ([num_heads] * num_layers) + [num_out_heads]

    with SummaryWriter('runs') as writer:
        print('Init model...')
        model = GAT(train_dataset.graph, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop, attn_drop, negative_slope, residual)
        print('Done')
        inputs = torch.randn((44906, 50))
        logits = model(inputs.float())
        writer.add_graph(model, inputs.float())
        # make_dot(logits).render("attached", format="png")
