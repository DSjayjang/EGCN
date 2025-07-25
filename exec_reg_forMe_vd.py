import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv_forMe_vd as mc

from model import EGCN_1_copy


from model import EGCN_3_copy
from model import EGCN_5_copy
from model import EGCN_7_copy
from model import EGCN_10_copy
from model import EGCN_20_copy

from model import Outer_EGCN_3_copy
from model import Outer_EGCN_5_copy
from model import Outer_EGCN_7_copy
from model import Outer_EGCN_10_copy
from model import Outer_EGCN_20_copy
from model import Outer_EGCN_elastic_copy

from util import trainer
from util import trainer_test

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
dataset_name = 'molproperty_vd'
batch_size = 32
max_epochs = 500
k = 5


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


"""
forMe 용
"""
########################################################################################################
def collate_emodel_elastic_1(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.ExactMolWt
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)



def collate_emodel_elastic_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.MolWt
        self_feats[i, 1] = mol_graph.fr_alkyl_halide
        self_feats[i, 2] = mol_graph.NumRotatableBonds
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_elastic_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.MolWt
        self_feats[i, 1] = mol_graph.fr_alkyl_halide
        self_feats[i, 2] = mol_graph.NumRotatableBonds
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.HeavyAtomMolWt
        ####################################################
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        

def collate_emodel_elastic_7(samples):
    self_feats = np.empty((len(samples), 6), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.MolWt
        self_feats[i, 1] = mol_graph.fr_alkyl_halide
        self_feats[i, 2] = mol_graph.NumRotatableBonds
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.HeavyAtomMolWt
        # 6
        self_feats[i, 5] = mol_graph.EState_VSA9
        # self_feats[i, 6] = mol_graph.BCUT2D_LOGPHI
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_elastic_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.MolWt
        self_feats[i, 1] = mol_graph.fr_alkyl_halide
        self_feats[i, 2] = mol_graph.NumRotatableBonds
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.HeavyAtomMolWt
        # 6
        self_feats[i, 5] = mol_graph.EState_VSA9
        self_feats[i, 6] = mol_graph.BCUT2D_LOGPHI
        self_feats[i, 7] = mol_graph.MinAbsEStateIndex
        self_feats[i, 8] = mol_graph.VSA_EState7
        self_feats[i, 9] = mol_graph.fr_allylic_oxid
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# def collate_emodel_elastic_20(samples):
#     self_feats = np.empty((len(samples), 20), dtype=np.float32)

#     for i in range(0, len(samples)):
#         mol_graph = samples[i][0]

#         ####################################################
#         # 1
#         self_feats[i, 0] = mol_graph.MolWt
#         self_feats[i, 1] = mol_graph.fr_alkyl_halide
#         self_feats[i, 2] = mol_graph.NumRotatableBonds
#         self_feats[i, 3] = mol_graph.FpDensityMorgan1
#         self_feats[i, 4] = mol_graph.HeavyAtomMolWt
#         # 6
#         self_feats[i, 5] = mol_graph.EState_VSA9
#         self_feats[i, 6] = mol_graph.BCUT2D_LOGPHI
#         self_feats[i, 7] = mol_graph.MinAbsEStateIndex
#         self_feats[i, 8] = mol_graph.VSA_EState7
#         self_feats[i, 9] = mol_graph.fr_allylic_oxid
#         # 11
#         self_feats[i, 10] = mol_graph.Chi3v
#         self_feats[i, 11] = mol_graph.PEOE_VSA8
#         self_feats[i, 12] = mol_graph.VSA_EState10
#         self_feats[i, 13] = mol_graph.MaxAbsPartialCharge
#         self_feats[i, 14] = mol_graph.PEOE_VSA4
#         # 16
#         self_feats[i, 15] = mol_graph.SlogP_VSA1
#         self_feats[i, 16] = mol_graph.fr_nitrile
#         self_feats[i, 17] = mol_graph.Chi4v
#         self_feats[i, 18] = mol_graph.MinPartialCharge
#         self_feats[i, 19] = mol_graph.PEOE_VSA13
#         ####################################################

#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
# #######################################################################################################


def collate_emodel_elastic(samples):
    self_feats = np.empty((len(samples), mc.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.MolWt
        self_feats[i, 1] = mol_graph.fr_alkyl_halide
        self_feats[i, 2] = mol_graph.NumRotatableBonds
        self_feats[i, 3] = mol_graph.FpDensityMorgan1
        self_feats[i, 4] = mol_graph.HeavyAtomMolWt
        # 6
        self_feats[i, 5] = mol_graph.EState_VSA9
        # self_feats[i, 6] = mol_graph.BCUT2D_LOGPHI
        self_feats[i, 6] = mol_graph.MinAbsEStateIndex
        self_feats[i, 7] = mol_graph.VSA_EState7
        self_feats[i, 8] = mol_graph.fr_allylic_oxid
        self_feats[i, 9] = mol_graph.Chi3v
        # 11
        self_feats[i, 10] = mol_graph.PEOE_VSA8
        self_feats[i, 11] = mol_graph.VSA_EState10
        # self_feats[i, 12] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 12] = mol_graph.PEOE_VSA4
        self_feats[i, 13] = mol_graph.SlogP_VSA1
        # 16
        self_feats[i, 14] = mol_graph.fr_nitrile
        self_feats[i, 15] = mol_graph.Chi4v
        # self_feats[i, 17] = mol_graph.MinPartialCharge
        self_feats[i, 16] = mol_graph.PEOE_VSA13
        self_feats[i, 17] = mol_graph.fr_nitro
        # 21
        self_feats[i, 18] = mol_graph.fr_ketone_Topliss
        self_feats[i, 19] = mol_graph.ExactMolWt
        # self_feats[i, 22] = mol_graph.
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
########################################################################################################


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = SEED)


#=====================================================================#
#=========================== Embedding : 1 ===========================#
#=====================================================================#

# # EGCN
model_EGCN_1 = EGCN_1_copy.Net(mc.dim_atomic_feat, 1, 1).to(device)

model_EGCN_3 = EGCN_3_copy.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_EGCN_5 = EGCN_5_copy.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_EGCN_7 = EGCN_7_copy.Net(mc.dim_atomic_feat, 1, 6).to(device)
model_EGCN_10 = EGCN_10_copy.Net(mc.dim_atomic_feat, 1, 10).to(device)
model_EGCN_20 = Outer_EGCN_elastic_copy.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Outer_EGCN
model_Outer_EGCN_3 = Outer_EGCN_3_copy.Net(mc.dim_atomic_feat, 1, 3).to(device)
model_Outer_EGCN_5 = Outer_EGCN_5_copy.Net(mc.dim_atomic_feat, 1, 5).to(device)
model_Outer_EGCN_7 = Outer_EGCN_7_copy.Net(mc.dim_atomic_feat, 1, 7).to(device)
model_Outer_EGCN_10 = Outer_EGCN_10_copy.Net(mc.dim_atomic_feat, 1, 10).to(device)
model_Outer_EGCN_20 = Outer_EGCN_20_copy.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Self_Feature
model_Outer_EGCN_elastic = Outer_EGCN_elastic_copy.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)


#=====================================================================#
#=========================== Embedding : 2 ===========================#
#=====================================================================#




# define loss function
criterion = nn.L1Loss(reduction='sum') # MAE
# criterion = nn.MSELoss() # MSE

# train and evaluate competitors
test_losses = dict()


#=====================================================================#
#=========================== Embedding : 1 ===========================#
#=====================================================================#

#------------------------ EGCN ------------------------#

# # feature 1개
# print('--------- EGCN_1 ---------')
# test_losses['EGCN_1'] = trainer.cross_validation(dataset, model_EGCN_1, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_1)
# print('test loss (EGCN_1): ' + str(test_losses['EGCN_1']))

# # feature 3개
# print('--------- EGCN_3 ---------')
# test_losses['EGCN_3'] = trainer.cross_validation(dataset, model_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_3)
# print('test loss (EGCN_3): ' + str(test_losses['EGCN_3']))

# # feature 5개
# print('--------- EGCN_5 ---------')
# test_losses['EGCN_5'] = trainer.cross_validation(dataset, model_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_5)
# print('test loss (EGCN_5): ' + str(test_losses['EGCN_5']))

# # feature 7개
# print('--------- EGCN_7 ---------')
# test_losses['EGCN_7'] = trainer.cross_validation(dataset, model_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_7)
# print('test loss (EGCN_7): ' + str(test_losses['EGCN_7']))

# # feature 10개
# print('--------- EGCN_10 ---------')
# test_losses['EGCN_10'] = trainer.cross_validation(dataset, model_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_10)
# print('test loss (EGCN_10): ' + str(test_losses['EGCN_10']))

# # feature 20개
# print('--------- EGCN_20 ---------')
# test_losses['EGCN_20'] = trainer.cross_validation(dataset, model_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_20)
# print('test loss (EGCN_20): ' + str(test_losses['EGCN_20']))

# print('--------- EGCN_3 ---------')
# test_losses['EGCN_3'], best_model, best_k = trainer_test.cross_validation(dataset, model_EGCN_3, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel_elastic_3)
# print('test loss (EGCN_3): ' + str(test_losses['EGCN_3']))


#------------------------ Outer EGCN ------------------------#

# # feature 3개
# print('--------- Outer EGCN_3 ---------')
# test_losses['Outer_EGCN_3'] = trainer.cross_validation(dataset, model_Outer_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_3)
# print('test loss (Outer_EGCN_3): ' + str(test_losses['Outer_EGCN_3']))

print('--------- Outer EGCN_3 ---------')
test_losses['Outer_EGCN_3'], best_model, best_k = trainer_test.cross_validation(train_dataset, model_Outer_EGCN_3, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel_elastic_3)
print('test loss (Outer_EGCN_3): ' + str(test_losses['Outer_EGCN_3']))

# # feature 5개
# print('--------- Outer EGCN_5 ---------')
# test_losses['Outer_EGCN_5'] = trainer.cross_validation(dataset, model_Outer_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_5)
# print('test loss (Outer_EGCN_5): ' + str(test_losses['Outer_EGCN_5']))

# # feature 7개
# print('--------- Outer EGCN_7 ---------')
# test_losses['Outer_EGCN_7'] = trainer.cross_validation(dataset, model_Outer_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_7)
# print('test loss (Outer_EGCN_7): ' + str(test_losses['Outer_EGCN_7']))

# # feature 10개
# print('--------- Outer EGCN_10 ---------')
# test_losses['Outer_EGCN_10'] = trainer.cross_validation(dataset, model_Outer_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_10)
# print('test loss (Outer_EGCN_10): ' + str(test_losses['Outer_EGCN_10']))

# # feature 20개
# print('--------- Outer EGCN_20 ---------')
# test_losses['Outer_EGCN_20'] = trainer.cross_validation(dataset, model_Outer_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
# print('test loss (Outer_EGCN_20): ' + str(test_losses['Outer_EGCN_20']))


#------------------------ Self Feature ------------------------#

# print('--------- Outer EGCN_elastic ---------')
# test_losses['Outer_EGCN_elastic'] = trainer.cross_validation(dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
# print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))

# print('--------- Outer EGCN_elastic ---------')
# test_losses['Outer_EGCN_elastic'], best_model, best_k = trainer_test.cross_validation(train_dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel_elastic)
# print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))

print(test_losses)

# # 최종 평가
# test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_emodel_elastic)
# final_test_loss, final_preds = trainer_test.final_test_emodel(best_model, criterion, test_data_loader)

# print(f'Final Test Loss: {final_test_loss}')
# #=====================================================================#
# #=========================== Embedding : 2 ===========================#
# #=====================================================================#

# print('best_k-fold:', best_k)
