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

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)
        self.fc1 = nn.Linear(20 * dim_self_feat, 10)
        self.fc2 = nn.Linear(10, dim_out)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h
        
        hg = dgl.mean_nodes(g, 'h')
        
        ## 함수로 정의하면 좋을 것 같음
        new_hg = []
        for i in range(hg.shape[0]):
            matmul_result = torch.mm(hg[i].unsqueeze(0).T, self_feat[i].unsqueeze(0))
            new_hg.append(matmul_result.flatten())
        hg = torch.stack(new_hg)

        out = F.relu(self.fc1(hg))
        out = self.fc2(out)

        return out
