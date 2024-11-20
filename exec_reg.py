import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv as mc
from model import GCN
from model import EGCN
from model import Extended_EGCN
from model import hjh_EGCN
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


# experiment parameters
dataset_name = 'logVP2'
batch_size = 32
max_epochs = 1
k = 2


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


def collate_emodel_scale(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)


def collate_emodel_ring(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)

"""
추가추가추가
반복반복반복
"""

# feature 3개
def collate_emodel_three_Extended(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)

def collate_emodel_three_newExtended(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.EState_VSA9
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.BCUT2D_LOGPLOW

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# feature 5개
def collate_emodel_five(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings
        self_feats[i, 3] = mol_graph.num_rad_elc
        self_feats[i, 4] = mol_graph.num_val_elc

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# feature 7개
def collate_emodel_seven(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings
        self_feats[i, 3] = mol_graph.num_rad_elc
        self_feats[i, 4] = mol_graph.num_val_elc
        self_feats[i, 5] = mol_graph.MolLogP
        self_feats[i, 6] = mol_graph.BertzCT

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# self feature
def collate_emodel(samples):
    self_feats = np.empty((len(samples), mc.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight
        self_feats[i, 2] = mol_graph.num_rings

        # 추가
        # self_feats[i, 3] = mol_graph.max_abs_charge
        # self_feats[i, 4] = mol_graph.min_abs_charge
        # self_feats[i, 5] = mol_graph.num_rad_elc
        # self_feats[i, 6] = mol_graph.num_val_elc
        # # 추가
        self_feats[i, 3] = mol_graph.num_rad_elc
        self_feats[i, 4] = mol_graph.num_val_elc

        # # 새로 추가
#        self_feats[i, 5] = mol_graph.NHOHCount # [logVP2] EGCN에서는 안돌아감
        self_feats[i, 5] = mol_graph.MolLogP
        self_feats[i, 6] = mol_graph.BertzCT
        self_feats[i, 7] = mol_graph.TPSA
        self_feats[i, 8] = mol_graph.fr_halogen
        self_feats[i, 9] = mol_graph.fr_amide
        # self_feats[i, 10] = mol_graph.MolLogP
        # self_feats[i, 11] = mol_graph.SMR_VSA10
        # self_feats[i, 12] = mol_graph.EState_VSA5
        # self_feats[i, 13] = mol_graph.SMR_VSA6
        # self_feats[i, 14] = mol_graph.EState_VSA1
        # self_feats[i, 15] = mol_graph.BCUT2D_LOGPHI
        # self_feats[i, 16] = mol_graph.VSA_EState8
        # self_feats[i, 17] = mol_graph.PEOE_VSA6
        # self_feats[i, 18] = mol_graph.VSA_EState9
        # self_feats[i, 19] = mol_graph.PEOE_VSA5
        # 새로 추가


    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)

# define model
#model_GCN = GCN.Net(mc.dim_atomic_feat, 1).to(device)
#model_EGCN_S = EGCN.Net(mc.dim_atomic_feat, 1, 2).to(device)
#model_EGCN_R = EGCN.Net(mc.dim_atomic_feat, 1, 1).to(device)

# feature 3개
model_EGCN_three = EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_Extended_EGCN_three = Extended_EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_hjh_EGCN_three = hjh_EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_test_EGCN_three = test_EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)

# feature 5개
model_EGCN_five = EGCN.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_Extended_EGCN_five = Extended_EGCN.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_hjh_EGCN_five = hjh_EGCN.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_test_EGCN_five = test_EGCN.Net(mc.dim_atomic_feat, 1, 5).to(device)

# feature 7개
model_EGCN_seven = EGCN.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_Extended_EGCN_seven = Extended_EGCN.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_hjh_EGCN_seven = hjh_EGCN.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_test_EGCN_seven = test_EGCN.Net(mc.dim_atomic_feat, 1, 7).to(device)

# self feature
model_EGCN = EGCN.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
model_Extended_EGCN = Extended_EGCN.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
model_hjh_EGCN = hjh_EGCN.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
model_test_EGCN = test_EGCN.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)


# define loss function
criterion = nn.L1Loss(reduction='sum')


# train and evaluate competitors
test_losses = dict()

# print('--------- GCN ---------')
# test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, k, batch_size, max_epochs, trainer.train, trainer.test, collate)
# print('test loss (GCN): ' + str(test_losses['GCN']))

# print('--------- EGCN_SCALE ---------')
# test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_scale)
# print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

# print('--------- EGCN_RING ---------')
# test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_ring)
# print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))
"""
추가추가추가
반복반복반복
"""

# feature 3개
print('--------- EGCN_three ---------')
test_losses['EGCN_three'] = trainer.cross_validation(dataset, model_EGCN_three, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_three_Extended)
print('test loss (EGCN_three): ' + str(test_losses['EGCN_three']))

print('--------- Exteded EGCN_three ---------')
test_losses['Extended_EGCN_three'] = trainer.cross_validation(dataset, model_Extended_EGCN_three, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_three_Extended)
print('test loss (Extended_EGCN_three): ' + str(test_losses['Extended_EGCN_three']))

print('---------new Exteded EGCN_three ---------')
test_losses['Extended_EGCN_three'] = trainer.cross_validation(dataset, model_Extended_EGCN_three, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_three_newExtended)
print('test loss (Extended_EGCN_three): ' + str(test_losses['Extended_EGCN_three']))

# print('--------- hjh EGCN_three ---------')
# test_losses['hjh_EGCN_three'] = trainer.cross_validation(dataset, model_hjh_EGCN_three, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_three)
# print('test loss (hjh_EGCN_three): ' + str(test_losses['hjh_EGCN_three']))

# print('--------- test EGCN_three ---------')
# test_losses['test_EGCN_three'] = trainer.cross_validation(dataset, model_test_EGCN_three, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_three)
# print('test loss (test_EGCN_three): ' + str(test_losses['test_EGCN_three']))


# feature 5개
# print('--------- EGCN_five ---------')
# test_losses['EGCN_five'] = trainer.cross_validation(dataset, model_EGCN_five, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_five)
# print('test loss (EGCN_five): ' + str(test_losses['EGCN_five']))

# print('--------- Exteded EGCN_five ---------')
# test_losses['Extended_EGCN_five'] = trainer.cross_validation(dataset, model_Extended_EGCN_five, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_five)
# print('test loss (Extended_EGCN_five): ' + str(test_losses['Extended_EGCN_five']))

# print('--------- hjh EGCN_five ---------')
# test_losses['hjh_EGCN_five'] = trainer.cross_validation(dataset, model_hjh_EGCN_five, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_five)
# print('test loss (hjh_EGCN_five): ' + str(test_losses['hjh_EGCN_five']))

# print('--------- test EGCN_five ---------')
# test_losses['test_EGCN_five'] = trainer.cross_validation(dataset, model_test_EGCN_five, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_five)
# print('test loss (test_EGCN_five): ' + str(test_losses['test_EGCN_five']))


# # feature 7개
# print('--------- EGCN_seven ---------')
# test_losses['EGCN_seven'] = trainer.cross_validation(dataset, model_EGCN_seven, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_seven)
# print('test loss (EGCN_seven): ' + str(test_losses['EGCN_seven']))

# print('--------- Exteded EGCN_seven ---------')
# test_losses['Extended_EGCN_seven'] = trainer.cross_validation(dataset, model_Extended_EGCN_seven, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_seven)
# print('test loss (Extended_EGCN_seven): ' + str(test_losses['Extended_EGCN_seven']))

# print('--------- hjh EGCN_seven ---------')
# test_losses['hjh_EGCN_seven'] = trainer.cross_validation(dataset, model_hjh_EGCN_seven, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_seven)
# print('test loss (hjh_EGCN_seven): ' + str(test_losses['hjh_EGCN_seven']))

# print('--------- test EGCN_seven ---------')
# test_losses['test_EGCN_seven'] = trainer.cross_validation(dataset, model_test_EGCN_seven, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_seven)
# print('test loss (test_EGCN_seven): ' + str(test_losses['test_EGCN_seven']))


# # self feature
# print('--------- EGCN ---------')
# test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
# print('test loss (EGCN): ' + str(test_losses['EGCN']))

# print('--------- Exteded EGCN ---------')
# test_losses['Extended_EGCN'] = trainer.cross_validation(dataset, model_Extended_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
# print('test loss (Extended_EGCN): ' + str(test_losses['Extended_EGCN']))

# print('--------- hjh EGCN ---------')
# test_losses['hjh_EGCN'] = trainer.cross_validation(dataset, model_hjh_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
# print('test loss (hjh_EGCN): ' + str(test_losses['hjh_EGCN']))

# print('--------- test EGCN ---------')
# test_losses['test_EGCN'] = trainer.cross_validation(dataset, model_test_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
# print('test loss (test_EGCN): ' + str(test_losses['test_EGCN']))

# 테스트
#print(f'Dimension of EGCN: {batch_size} x {model_EGCN.fc1.in_features}')
#print(f'Dimension of Extended_EGCN: {batch_size} x {model_Extended_EGCN.fc1.in_features}')

print(test_losses)
