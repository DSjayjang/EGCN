import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv_logvp28_cheerypicker as mc
import copy
import torch.optim as optim


from model import EGCN_3
from model import EGCN_5
from model import EGCN_7
from model import EGCN_10
from model import EGCN_20
from model import EGCN_elastic

from model import Outer_EGCN_3
from model import Outer_EGCN_5
from model import Outer_EGCN_7
from model import Outer_EGCN_10
from model import Outer_EGCN_20
from model import Outer_EGCN_elastic

from util import trainer
from util import trainer_test_real_cherrypicker

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# 재현성-난수 고정
import os
#import tensorflow as tf

SEED = 100

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
dataset_name = 'logvp_cp'
batch_size = 128
max_epochs = 1
k = 2


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


########################################################################################################
########################################################################################################
# def collate_emodel_elastic_1(samples):
#     self_feats = np.empty((len(samples), 1), dtype=np.float32)

#     for i in range(0, len(samples)):
#         mol_graph = samples[i][0]

#         ####################################################
#         # 1
#         self_feats[i, 0] = mol_graph.BCUT2D_CHGLO
#         ####################################################

#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_elastic_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.TPSA
        self_feats[i, 1] = mol_graph.MolMR
        self_feats[i, 2] = mol_graph.Chi1
        ####################################################

    graphs, labels, smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device), smiles


def collate_emodel_elastic_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.TPSA
        self_feats[i, 1] = mol_graph.MolMR
        self_feats[i, 2] = mol_graph.Chi1
        self_feats[i, 3] = mol_graph.NumHAcceptors
        self_feats[i, 4] = mol_graph.NumAromaticRings
        ####################################################
    graphs, labels, smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device), smiles
        

def collate_emodel_elastic_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.TPSA
        self_feats[i, 1] = mol_graph.MolMR
        self_feats[i, 2] = mol_graph.Chi1
        self_feats[i, 3] = mol_graph.NumHAcceptors
        self_feats[i, 4] = mol_graph.NumAromaticRings
        # 6
        self_feats[i, 5] = mol_graph.NOCount
        self_feats[i, 6] = mol_graph.SlogP_VSA12
        ####################################################

    graphs, labels, smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device), smiles


def collate_emodel_elastic_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.TPSA
        self_feats[i, 1] = mol_graph.MolMR
        self_feats[i, 2] = mol_graph.Chi1
        self_feats[i, 3] = mol_graph.NumHAcceptors
        self_feats[i, 4] = mol_graph.NumAromaticRings
        # 6
        self_feats[i, 5] = mol_graph.NOCount
        self_feats[i, 6] = mol_graph.SlogP_VSA12
        self_feats[i, 7] = mol_graph.RingCount
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.fr_halogen
        ####################################################

    graphs, labels, smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device), smiles


def collate_emodel_elastic_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.TPSA
        self_feats[i, 1] = mol_graph.MolMR
        self_feats[i, 2] = mol_graph.Chi1
        self_feats[i, 3] = mol_graph.NumHAcceptors
        self_feats[i, 4] = mol_graph.NumAromaticRings
        # 6
        self_feats[i, 5] = mol_graph.NOCount
        self_feats[i, 6] = mol_graph.SlogP_VSA12
        self_feats[i, 7] = mol_graph.RingCount
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.fr_halogen
        # 11
        self_feats[i, 10] = mol_graph.Chi2n
        self_feats[i, 11] = mol_graph.NumHeteroatoms
        self_feats[i, 12] = mol_graph.Chi1v
        self_feats[i, 13] = mol_graph.SMR_VSA10
        self_feats[i, 14] = mol_graph.VSA_EState1
        # 16
        self_feats[i, 15] = mol_graph.VSA_EState6
        self_feats[i, 16] = mol_graph.MolLogP
        self_feats[i, 17] = mol_graph.Chi4n
        self_feats[i, 18] = mol_graph.Chi0v
        self_feats[i, 19] = mol_graph.BertzCT
        ####################################################

    graphs, labels, smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device), smiles
########################################################################################################


def collate_emodel_elastic(samples):
    self_feats = np.empty((len(samples), mc.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.TPSA
        self_feats[i, 1] = mol_graph.MolMR
        self_feats[i, 2] = mol_graph.Chi1
        self_feats[i, 3] = mol_graph.NumHAcceptors
        self_feats[i, 4] = mol_graph.NumAromaticRings
        # 6
        self_feats[i, 5] = mol_graph.NOCount
        self_feats[i, 6] = mol_graph.SlogP_VSA12
        self_feats[i, 7] = mol_graph.RingCount
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.fr_halogen
        # 11
        self_feats[i, 10] = mol_graph.Chi2n
        self_feats[i, 11] = mol_graph.NumHeteroatoms
        self_feats[i, 12] = mol_graph.Chi1v
        self_feats[i, 13] = mol_graph.SMR_VSA10
        self_feats[i, 14] = mol_graph.VSA_EState1
        # 16
        self_feats[i, 15] = mol_graph.VSA_EState6
        self_feats[i, 16] = mol_graph.MolLogP
        self_feats[i, 17] = mol_graph.Chi4n
        self_feats[i, 18] = mol_graph.Chi0v
        self_feats[i, 19] = mol_graph.BertzCT
        # 21
        self_feats[i, 20] = mol_graph.HallKierAlpha
        ####################################################

    graphs, labels, smiles = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device), smiles
########################################################################################################


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = SEED)


#=====================================================================#
#=========================== Embedding : 1 ===========================#
#=====================================================================#

# EGCN
# model_EGCN_1 = Outer_EGCN_elastic.Net(mc.dim_atomic_feat, 1, 1).to(device)

model_EGCN_3 = EGCN_3.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_EGCN_5 = EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_EGCN_7 = EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_EGCN_10 = EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
model_EGCN_20 = EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Outer_EGCN
model_Outer_EGCN_3 = Outer_EGCN_3.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_Outer_EGCN_5 = Outer_EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_Outer_EGCN_7 = Outer_EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_Outer_EGCN_10 = Outer_EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
model_Outer_EGCN_20 = Outer_EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Self_Feature
model_EGCN_elastic = EGCN_elastic.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
model_Outer_EGCN_elastic = Outer_EGCN_elastic.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)


#=====================================================================#
#=========================== Embedding : 2 ===========================#
#=====================================================================#




# define loss function
# criterion = nn.L1Loss(reduction='sum') # MAE
criterion = nn.MSELoss(reduction='sum') # MSE

# train and evaluate competitors
val_losses = dict()
test_losses = dict()


#=====================================================================#
#=========================== Embedding : 1 ===========================#
#=====================================================================#

#------------------------ EGCN ------------------------#

# # feature 1개
# print('--------- EGCN_1 ---------')
# test_losses['EGCN_1'] = trainer.cross_validation(dataset, model_EGCN_1, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_1)
# print('test loss (EGCN_1): ' + str(test_losses['EGCN_1']))

# feature 3개
print('--------- EGCN_3 ---------')
test_losses['EGCN_3'] = trainer.cross_validation(dataset, model_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_3)
print('test loss (EGCN_3): ' + str(test_losses['EGCN_3']))

# feature 5개
print('--------- EGCN_5 ---------')
test_losses['EGCN_5'] = trainer.cross_validation(dataset, model_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_5)
print('test loss (EGCN_5): ' + str(test_losses['EGCN_5']))

# feature 7개
print('--------- EGCN_7 ---------')
test_losses['EGCN_7'] = trainer.cross_validation(dataset, model_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_7)
print('test loss (EGCN_7): ' + str(test_losses['EGCN_7']))

# feature 10개
print('--------- EGCN_10 ---------')
test_losses['EGCN_10'] = trainer.cross_validation(dataset, model_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_10)
print('test loss (EGCN_10): ' + str(test_losses['EGCN_10']))

# feature 20개
print('--------- EGCN_20 ---------')
test_losses['EGCN_20'] = trainer.cross_validation(dataset, model_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_20)
print('test loss (EGCN_20): ' + str(test_losses['EGCN_20']))

# feature 20개
print('--------- EGCN_elastic ---------')
test_losses['EGCN_elastic'] = trainer.cross_validation(dataset, model_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
print('test loss (EGCN_elastic): ' + str(test_losses['EGCN_elastic']))

#------------------------ Outer EGCN ------------------------#

# feature 3개
print('--------- Outer EGCN_3 ---------')
test_losses['Outer_EGCN_3'] = trainer.cross_validation(dataset, model_Outer_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_3)
print('test loss (Outer_EGCN_3): ' + str(test_losses['Outer_EGCN_3']))

# feature 5개
print('--------- Outer EGCN_5 ---------')
test_losses['Outer_EGCN_5'] = trainer.cross_validation(dataset, model_Outer_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_5)
print('test loss (Outer_EGCN_5): ' + str(test_losses['Outer_EGCN_5']))

# feature 7개
print('--------- Outer EGCN_7 ---------')
test_losses['Outer_EGCN_7'] = trainer.cross_validation(dataset, model_Outer_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_7)
print('test loss (Outer_EGCN_7): ' + str(test_losses['Outer_EGCN_7']))

# feature 10개
print('--------- Outer EGCN_10 ---------')
test_losses['Outer_EGCN_10'] = trainer.cross_validation(dataset, model_Outer_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_10)
print('test loss (Outer_EGCN_10): ' + str(test_losses['Outer_EGCN_10']))

# feature 20개
print('--------- Outer EGCN_20 ---------')
test_losses['Outer_EGCN_20'] = trainer.cross_validation(dataset, model_Outer_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_20)
print('test loss (Outer_EGCN_20): ' + str(test_losses['Outer_EGCN_20']))

print('--------- Outer EGCN_elastic ---------')
test_losses['Outer_EGCN_elastic'] = trainer.cross_validation(dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))
print('test_losse:', test_losses)

print(test_losses)

# #------------------------ Self Feature ------------------------#

# print('--------- Outer EGCN_elastic ---------')
# test_losses['Outer_EGCN_elastic'] = trainer_test.cross_validation(dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel_elastic)
# print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))




# # 최종 평가
# print('--------- Outer EGCN_elastic ---------')
# val_losses['Outer_EGCN_elastic'], best_model, best_k = trainer_test_real_cherrypicker.cross_validation(train_dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer_test_real_cherrypicker.train_model, trainer_test_real_cherrypicker.val_model, collate_emodel_elastic)
# print('Val loss (Outer_EGCN_elastic): ' + str(val_losses['Outer_EGCN_elastic']))

# final_model = copy.deepcopy(best_model)
# def weight_reset(m):
#     if hasattr(m, 'reset_parameters'):
#         m.reset_parameters()

# final_model.apply(weight_reset)


# optimizer = optim.Adam(final_model.parameters(), weight_decay=0.01)

# # 전체 트레이닝용 dataset
# train_data_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_emodel_elastic)
# final_train_loss = trainer_test_real_cherrypicker.train_model(final_model, criterion, optimizer, train_data_loader, max_epochs)

# # 트레이닝 평가용
# trainer_test_real_cherrypicker.collect_train_preds(final_model, criterion, train_data_loader)

# # final test
# test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_emodel_elastic)
# test_loss, final_preds = trainer_test_real_cherrypicker.test_model(final_model, criterion, test_data_loader)

# print('best_k-fold:', best_k)
# print('after k-fold, averaging of val_losses:', val_losses)
# print('test_losse:', test_loss)