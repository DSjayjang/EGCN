import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv_qm7 as mc
from model import GCN

from model import EGCN
from model import EGCN_3
from model import EGCN_5
from model import EGCN_7
from model import EGCN_10
from model import EGCN_20

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
dgl.random.seed(SEED)
#tf.random.set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# experiment parameters
dataset_name = 'qm7_reduced'
batch_size = 32
max_epochs = 300
k = 5


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


"""
qm7_reduced
"""
########################################################################################################
# qm7_reduced
# don't touch
def collate_emodel_Extended_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # qm7_reduced
        # 3
        self_feats[i, 0] = mol_graph.Chi1n
        self_feats[i, 1] = mol_graph.Chi1v
        self_feats[i, 2] = mol_graph.Chi0v
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # qm7_reduced
        # 3
        self_feats[i, 0] = mol_graph.Chi1n
        self_feats[i, 1] = mol_graph.Chi1v
        self_feats[i, 2] = mol_graph.Chi0v
        # 5
        self_feats[i, 3] = mol_graph.MaxPartialCharge
        self_feats[i, 4] = mol_graph.Kappa2
        ####################################################
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        

def collate_emodel_Extended_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # qm7_reduced
        # 3
        self_feats[i, 0] = mol_graph.Chi1n
        self_feats[i, 1] = mol_graph.Chi1v
        self_feats[i, 2] = mol_graph.Chi0v
        # 5
        self_feats[i, 3] = mol_graph.MaxPartialCharge
        self_feats[i, 4] = mol_graph.Kappa2
        # 7
        self_feats[i, 5] = mol_graph.SlogP_VSA4
        self_feats[i, 6] = mol_graph.EState_VSA9
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # qm7_reduced
        # 3
        self_feats[i, 0] = mol_graph.Chi1n
        self_feats[i, 1] = mol_graph.Chi1v
        self_feats[i, 2] = mol_graph.Chi0v
        # 5
        self_feats[i, 3] = mol_graph.MaxPartialCharge
        self_feats[i, 4] = mol_graph.Kappa2
        # 7
        self_feats[i, 5] = mol_graph.SlogP_VSA4
        self_feats[i, 6] = mol_graph.EState_VSA9
        # 10
        self_feats[i, 7] = mol_graph.VSA_EState2
        self_feats[i, 8] = mol_graph.PEOE_VSA4
        self_feats[i, 9] = mol_graph.SlogP_VSA3
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # qm7_reduced
        # 3
        self_feats[i, 0] = mol_graph.Chi1n
        self_feats[i, 1] = mol_graph.Chi1v
        self_feats[i, 2] = mol_graph.Chi0v
        # 5
        self_feats[i, 3] = mol_graph.MaxPartialCharge
        self_feats[i, 4] = mol_graph.Kappa2
        # 7
        self_feats[i, 5] = mol_graph.SlogP_VSA4
        self_feats[i, 6] = mol_graph.EState_VSA9
        # 10
        self_feats[i, 7] = mol_graph.VSA_EState2
        self_feats[i, 8] = mol_graph.PEOE_VSA4
        self_feats[i, 9] = mol_graph.SlogP_VSA3
        # 20
        self_feats[i, 10] = mol_graph.BCUT2D_MRLOW
        self_feats[i, 11] = mol_graph.PEOE_VSA11
        self_feats[i, 12] = mol_graph.SlogP_VSA8
        self_feats[i, 13] = mol_graph.EState_VSA7
        self_feats[i, 14] = mol_graph.fr_Ndealkylation1
        self_feats[i, 15] = mol_graph.SMR_VSA5
        self_feats[i, 16] = mol_graph.fr_epoxide
        self_feats[i, 17] = mol_graph.fr_NH1
        self_feats[i, 18] = mol_graph.VSA_EState7
        self_feats[i, 19] = mol_graph.qed
        ####################################################


    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
########################################################################################################


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)


# EGCN
model_EGCN_3 = EGCN_3.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_EGCN_5 = EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_EGCN_7 = EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_EGCN_10 = EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
model_EGCN_20 = EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Extended_EGCN
model_Extended_EGCN_3 = Extended_EGCN_3.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_Extended_EGCN_5 = Extended_EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_Extended_EGCN_7 = Extended_EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_Extended_EGCN_10 = Extended_EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
model_Extended_EGCN_20 = Extended_EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Bilinear_EGCN
# model_Bilinear_EGCN_3 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
# model_Bilinear_EGCN_5 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 5).to(device)
# model_Bilinear_EGCN_7 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 7).to(device)
# model_Bilinear_EGCN_10 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 10).to(device)
# model_Bilinear_EGCN_20 = Bilinear_EGCN.Net(mc.dim_atomic_feat, 1, 20).to(device)

# define loss function
criterion = nn.L1Loss(reduction='sum')

# train and evaluate competitors
test_losses = dict()


# self feature

# feature 3개
print('--------- EGCN_3 ---------')
test_losses['EGCN_3'] = trainer.cross_validation(dataset, model_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_3)
print('test loss (EGCN_3): ' + str(test_losses['EGCN_3']))

# feature 5개
print('--------- EGCN_5 ---------')
test_losses['EGCN_5'] = trainer.cross_validation(dataset, model_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_5)
print('test loss (EGCN_5): ' + str(test_losses['EGCN_5']))

# feature 7개
print('--------- EGCN_7 ---------')
test_losses['EGCN_7'] = trainer.cross_validation(dataset, model_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_7)
print('test loss (EGCN_7): ' + str(test_losses['EGCN_7']))

# feature 10개
print('--------- EGCN_10 ---------')
test_losses['EGCN_10'] = trainer.cross_validation(dataset, model_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_10)
print('test loss (EGCN_10): ' + str(test_losses['EGCN_10']))

# feature 20개
print('--------- EGCN_20 ---------')
test_losses['EGCN_20'] = trainer.cross_validation(dataset, model_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_20)
print('test loss (EGCN_20): ' + str(test_losses['EGCN_20']))


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


# asd
# asd
