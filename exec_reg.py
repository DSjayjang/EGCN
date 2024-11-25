import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv as mc
from model import GCN
from model import EGCN
from model import Extended_EGCN
from model import Bilinear_EGCN
from model import test_EGCN
from util import trainer

# 재현성-난수 고정
import os
#import tensorflow as tf

SEED = 100

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
#tf.random.set_seed(SEED)


# check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# experiment parameters
dataset_name = 'esol'
batch_size = 32
max_epochs = 300
k = 5


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

# ring 1개
def collate_emodel_ring(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)

# num_atoms + weight 2개
def collate_emodel_scale(samples):
    self_feats = np.empty((len(samples), 2), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)


# num_atoms + weight + ring 3개
def collate_emodel(samples):
    self_feats = np.empty((len(samples), mc.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

# self feature
def collate_emodel_Extended(samples):
    self_feats = np.empty((len(samples), mc.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.HeavyAtomCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.BertzCT
        self_feats[i, 3] = mol_graph.fr_C_O_noCOO
        self_feats[i, 4] = mol_graph.NumSaturatedCarbocycles
        self_feats[i, 5] = mol_graph.EState_VSA10
        self_feats[i, 6] = mol_graph.MinPartialCharge
        self_feats[i, 7] = mol_graph.TPSA
        self_feats[i, 8] = mol_graph.MolMR


    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)



# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)

# default model
# don't touch
model_GCN = GCN.Net(mc.dim_atomic_feat, 1).to(device)
model_EGCN_R = EGCN.Net(mc.dim_atomic_feat, 1, 1).to(device)
model_EGCN_S = EGCN.Net(mc.dim_atomic_feat, 1, 2).to(device)
model_EGCN = EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)

# self feature
model_Extended_EGCN = Extended_EGCN.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
model_Bilinear_EGCN = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)

# define loss function
criterion = nn.L1Loss(reduction='sum')

# train and evaluate competitors
test_losses = dict()


#=====================================================================#
# default model
# don't touch

# print('--------- GCN ---------')
# test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, k, batch_size, max_epochs, trainer.train, trainer.test, collate)
# print('test loss (GCN): ' + str(test_losses['GCN']))

# print('--------- EGCN_RING ---------')
# test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_ring)
# print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

# print('--------- EGCN_SCALE ---------')
# test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_scale)
# print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

# print('--------- EGCN ---------')
# test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
# print('test loss (EGCN): ' + str(test_losses['EGCN']))
#=====================================================================#


# self feature

print('--------- Exteded EGCN ---------')
test_losses['Extended_EGCN'] = trainer.cross_validation(dataset, model_Extended_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
print('test loss (Extended_EGCN): ' + str(test_losses['Extended_EGCN']))

print('--------- Bilinear EGCN ---------')
test_losses['Bilinear_EGCN'] = trainer.cross_validation(dataset, model_Bilinear_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
print('test loss (Bilinear_EGCN): ' + str(test_losses['Bilinear_EGCN']))

# print('--------- test EGCN ---------')
# test_losses['test_EGCN'] = trainer.cross_validation(dataset, model_test_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
# print('test loss (test_EGCN): ' + str(test_losses['test_EGCN']))

# 테스트
#print(f'Dimension of EGCN: {batch_size} x {model_EGCN.fc1.in_features}')
#print(f'Dimension of Extended_EGCN: {batch_size} x {model_Extended_EGCN.fc1.in_features}')

print(test_losses)
