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


# class Net(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_self_feat):
#         super(Net, self).__init__()

#         self.gc1 = GCNLayer(dim_in, 100)
#         self.gc2 = GCNLayer(100, 20)

#         self.fc1 = nn.Linear(20 * dim_self_feat, 256)
#         self.fc2 = nn.Linear(256, 32)
#         self.fc3 = nn.Linear(32, dim_out)

#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(32)
#         self.bn3 = nn.BatchNorm1d(8)
#         self.dropout = nn.Dropout(0.3)


#     def forward(self, g, self_feat):
#         # 그래프 합성곱
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))
#         g.ndata['h'] = h

#         # 그래프 임베딩 생성
#         hg = dgl.mean_nodes(g, 'h')

#         # 통합
#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         hg = torch.bmm(hg, self_feat)
#         hg = hg.view(hg.size(0), -1)

#         # FCNN
#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)

#         out = F.relu(self.bn2(self.fc2(out)))

#         out = self.fc3(out)

#         return out


class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)

        self.fc1 = nn.Linear(20 * dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)
        # self.fc3 = nn.Linear(4, dim_out)
        # self.fc3 = nn.Linear(16, dim_out)

        self.bn1 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(0.2)


    def forward(self, g, self_feat):
        # 그래프 합성곱
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        # 그래프 임베딩 생성
        hg = dgl.mean_nodes(g, 'h')

        # 통합
        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        # FCNN
        # out = F.relu(self.bn1(self.fc1(hg)))
        # out = self.dropout(out)

        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out