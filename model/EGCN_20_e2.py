import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


#msg = fn.copy_src(src='h', out='m')
msg = fn.copy_u('h', 'm')

def reduce(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)

    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, node):
        h = self.linear(node.data['h'])

        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(dim_in, dim_out)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)

        return g.ndata.pop('h')


class Net(nn.Module):  
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(Net, self).__init__()
        split_idx = dim_self_feat // 2 + dim_self_feat % 2
        self.split_idx = split_idx

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        # self.self_feat_fc = nn.Linear(dim_self_feat, 100)  # dim_self_feat → 100

        self.fc1 = nn.Linear(20 + dim_self_feat, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256 + dim_self_feat, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, dim_out)
        self.dropout = nn.Dropout(0.3)

    def combine_with_self_feat(self, h, self_feat, batch_num_nodes):
        self_feat_transformed = self.self_feat_fc(self_feat)  # (batch_size, 100)

        self_feat_expanded = torch.cat(
            [self_feat_transformed[i].repeat(n, 1) for i, n in enumerate(batch_num_nodes)], dim=0
        )  # (total_nodes_in_batch, 100)

        h = h + self_feat_expanded
        return h

    def forward(self, g, self_feat):
        batch_num_nodes = g.batch_num_nodes()

        # GCN Layer 1
        h = F.relu(self.gc1(g, g.ndata['feat']))
        # h = self.combine_with_self_feat(h, self_feat, batch_num_nodes)

        # GCN Layer 2
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        # self_feat_1 = self_feat[:, :self.split_idx]
        # self_feat_2 = self_feat[:, self.split_idx:]

        combined_local = torch.cat((hg, self_feat), dim=1)
        out = F.relu(self.bn1(self.fc1(combined_local)))
        out = self.dropout(out)

        combined_global = torch.cat((out, self_feat), dim=1)
        out = F.relu(self.bn2(self.fc2(combined_global)))
        out = self.fc3(out)

        return out
