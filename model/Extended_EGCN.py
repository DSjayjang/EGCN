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

#         self.fc1 = nn.Linear(20 * dim_self_feat, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, dim_out)

#         # batch normalization 추가
#         self.bn1 = nn.BatchNorm1d(128) # for the output of fc1
#         self.bn2 = nn.BatchNorm1d(64) # for the output of fc2
#         self.bn3 = nn.BatchNorm1d(32) # for the output of fc3

#         # droupout 추가
#         self.dropout = nn.Dropout(p = 0.5)

#     def forward(self, g, self_feat):
#         h = F.relu(self.gc1(g, g.ndata['feat']))
#         h = F.relu(self.gc2(g, h))
#         g.ndata['h'] = h
        
#         hg = dgl.mean_nodes(g, 'h')
        
#         ## 함수로 정의하면 좋을 것 같음
#         # new_hg = []
#         # for i in range(hg.shape[0]):
#         #     matmul_result = torch.mm(hg[i].unsqueeze(0).T, self_feat[i].unsqueeze(0))
#         #     new_hg.append(matmul_result.flatten())
#         # hg = torch.stack(new_hg)
# #        print('★★★★★ hg:', hg.shape)

#         hg = hg.unsqueeze(2)
#         self_feat = self_feat.unsqueeze(1)
#         """
#         torch.bmm()은 두 개의 3차원 텐서를 입력받음
#         이 텐서들은 각각 (b, n, m)과 (b, m, p)의 형태
#         b는 배치 크기, n, m, p는 각 행렬의 차원을 나타냅니다.
#         출력: (b, n, p) 형태의 3차원 텐서를 출력
#         """
#         hg = torch.bmm(hg, self_feat) 
#         hg = hg.view(hg.size(0), -1)



#         out = F.relu(self.bn1(self.fc1(hg)))
#         out = self.dropout(out)

#         out = F.relu(self.bn2(self.fc2(out)))
#         out = self.dropout(out)

#         out = F.relu(self.bn3(self.fc3(out)))
#         out = self.dropout(out)

#         out = self.fc4(out)

#         return out
class Net(nn.Module):
    def __init__(self, dim_in, dim_out, dim_self_feat):
        super(Net, self).__init__()

        self.gc1 = GCNLayer(dim_in, 100)
        self.gc2 = GCNLayer(100, 20)
        self.fc1 = nn.Linear(20 * dim_self_feat, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, dim_out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, g, self_feat):
        h = F.relu(self.gc1(g, g.ndata['feat']))
        h = F.relu(self.gc2(g, h))
        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        hg = hg.unsqueeze(2)
        self_feat = self_feat.unsqueeze(1)
        hg = torch.bmm(hg, self_feat)
        hg = hg.view(hg.size(0), -1)

        out = F.relu(self.bn1(self.fc1(hg)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)

        return out