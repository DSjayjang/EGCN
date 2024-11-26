import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv as mc
from model import GCN
from model import EGCN
from model import EGCN2
from model import Extended_EGCN
from model import Extended_EGCN_3
from model import Extended_EGCN_5
from model import Extended_EGCN_7
from model import Extended_EGCN_10
from model import Extended_EGCN_20
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
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# self feature
def collate_emodel_Extended_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 3
        self_feats[i, 0] = mol_graph.MinPartialCharge
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.MolMR

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 3
        self_feats[i, 0] = mol_graph.MinPartialCharge
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.MolMR
        # 5
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.SlogP_VSA6
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        

def collate_emodel_Extended_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 3
        self_feats[i, 0] = mol_graph.MinPartialCharge
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.MolMR
        # 5
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.SlogP_VSA6
        # 7
        self_feats[i, 5] = mol_graph.SMR_VSA5
        self_feats[i, 6] = mol_graph.HeavyAtomCount

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 3
        self_feats[i, 0] = mol_graph.MinPartialCharge
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.MolMR
        # 5
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.SlogP_VSA6
        # 7
        self_feats[i, 5] = mol_graph.SMR_VSA5
        self_feats[i, 6] = mol_graph.HeavyAtomCount
        # 10
        self_feats[i, 7] = mol_graph.FpDensityMorgan3
        self_feats[i, 8] = mol_graph.NumHAcceptors
        self_feats[i, 9] = mol_graph.RingCount

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        # 3
        self_feats[i, 0] = mol_graph.MinPartialCharge
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.MolMR
        # 5
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.SlogP_VSA6
        # 7
        self_feats[i, 5] = mol_graph.SMR_VSA5
        self_feats[i, 6] = mol_graph.HeavyAtomCount
        # 10
        self_feats[i, 7] = mol_graph.FpDensityMorgan3
        self_feats[i, 8] = mol_graph.NumHAcceptors
        self_feats[i, 9] = mol_graph.RingCount
        # 20
        self_feats[i, 10] = mol_graph.BCUT2D_CHGHI
        self_feats[i, 11] = mol_graph.EState_VSA9
        self_feats[i, 12] = mol_graph.FpDensityMorgan2
        self_feats[i, 13] = mol_graph.MinAbsPartialCharge
        self_feats[i, 14] = mol_graph.MinEStateIndex
        self_feats[i, 15] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 16] = mol_graph.SlogP_VSA5
        self_feats[i, 17] = mol_graph.SlogP_VSA8
        self_feats[i, 18] = mol_graph.VSA_EState7
        self_feats[i, 19] = mol_graph.fr_C_O_noCOO



    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)



# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)

#=====================================================================#
# default model
# don't touch
# model_GCN = GCN.Net(mc.dim_atomic_feat, 1).to(device)
# model_EGCN_R = EGCN.Net(mc.dim_atomic_feat, 1, 1).to(device)
# model_EGCN_S = EGCN.Net(mc.dim_atomic_feat, 1, 2).to(device)
# model_EGCN = EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
#=====================================================================#


# self feature
# model_EGCN2 = EGCN2.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)

# feature 3개
model_Extended_EGCN_3 = Extended_EGCN_3.Net(mc.dim_atomic_feat, 1, 3).to(device)
# model_Bilinear_EGCN_3 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
# feature 5개
model_Extended_EGCN_5 = Extended_EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
# model_Bilinear_EGCN_5 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 5).to(device)
# feature 7개
model_Extended_EGCN_7 = Extended_EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
# model_Bilinear_EGCN_7 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 7).to(device)
# feature 10개
model_Extended_EGCN_10 = Extended_EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
# model_Bilinear_EGCN_10 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 10).to(device)
# feature 20개
model_Extended_EGCN_20 = Extended_EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)
# model_Bilinear_EGCN_20 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 20).to(device)

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

# print('--------- EGCN2 ---------')
# test_losses['EGCN2'] = trainer.cross_validation(dataset, model_EGCN2, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
# print('test loss (EGCN2): ' + str(test_losses['EGCN2']))

# Extended_EGCN
# feature 3개
print('--------- Exteded EGCN_3 ---------')
test_losses['Extended_EGCN_3'] = trainer.cross_validation(dataset, model_Extended_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_3)
print('test loss (Extended_EGCN_3): ' + str(test_losses['Extended_EGCN_3']))

# feature 5개
print('--------- Exteded EGCN_5 ---------')
test_losses['Extended_EGCN_5'] = trainer.cross_validation(dataset, model_Extended_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_5)
print('test loss (Extended_EGCN_5): ' + str(test_losses['Extended_EGCN_5']))

# feature 7개
print('--------- Exteded EGCN_7 ---------')
test_losses['Extended_EGCN_7'] = trainer.cross_validation(dataset, model_Extended_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_7)
print('test loss (Extended_EGCN_7): ' + str(test_losses['Extended_EGCN_7']))

# feature 10개
print('--------- Exteded EGCN_10 ---------')
test_losses['Extended_EGCN_10'] = trainer.cross_validation(dataset, model_Extended_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_10)
print('test loss (Extended_EGCN_10): ' + str(test_losses['Extended_EGCN_10']))

# feature 20개
print('--------- Exteded EGCN_20 ---------')
test_losses['Extended_EGCN_20'] = trainer.cross_validation(dataset, model_Extended_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_20)
print('test loss (Extended_EGCN_20): ' + str(test_losses['Extended_EGCN_20']))


#=====================================================================#
# Bilinear_EGCN
# # feature 3개
# print('--------- Bilinear EGCN_3 ---------')
# test_losses['Bilinear_EGCN_3'] = trainer.cross_validation(dataset, model_Bilinear_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
# print('test loss (Bilinear_EGCN_3): ' + str(test_losses['Bilinear_EGCN3']))

# # feature 5개
# print('--------- Bilinear EGCN_5 ---------')
# test_losses['Bilinear_EGCN_5'] = trainer.cross_validation(dataset, model_Bilinear_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
# print('test loss (Bilinear_EGCN_5): ' + str(test_losses['Bilinear_EGCN_5']))

# # feature 7개
# print('--------- Bilinear EGCN_7 ---------')
# test_losses['Bilinear_EGCN_7'] = trainer.cross_validation(dataset, model_Bilinear_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
# print('test loss (Bilinear_EGCN_7): ' + str(test_losses['Bilinear_EGCN_7']))

# # feature 10개
# print('--------- Bilinear EGCN_10 ---------')
# test_losses['Bilinear_EGCN_10'] = trainer.cross_validation(dataset, model_Bilinear_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
# print('test loss (Bilinear_EGCN_10): ' + str(test_losses['Bilinear_EGCN_10']))

# # feature 20개
# print('--------- Bilinear EGCN_20 ---------')
# test_losses['Bilinear_EGCN_20'] = trainer.cross_validation(dataset, model_Bilinear_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended)
# print('test loss (Bilinear_EGCN_20): ' + str(test_losses['Bilinear_EGCN_20']))

print(test_losses)
