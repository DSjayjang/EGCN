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