import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv_forMe_vp as mc

from model import EGCN_1_copy
from model import EGCN_3
from model import EGCN_5
from model import EGCN_7
from model import EGCN_10
from model import EGCN_20
from model import EGCN_elastic

from model import Outer_EGCN_3_copy
from model import Outer_EGCN_5
from model import Outer_EGCN_7
from model import Outer_EGCN_10
from model import Outer_EGCN_20
from model import Outer_EGCN_elastic_copy2

from util import trainer
from util import trainer_test

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
dataset_name = 'molproperty_vp'
batch_size = 32
max_epochs = 300
k = 5


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


"""
esol 용
"""
########################################################################################################
def collate_emodel_elastic_1(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.Chi0n
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
        self_feats[i, 0] = mol_graph.NOCount
        self_feats[i, 1] = mol_graph.SlogP_VSA1
        self_feats[i, 2] = mol_graph.LabuteASA
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# def collate_emodel_elastic_5(samples):
#     self_feats = np.empty((len(samples), 5), dtype=np.float32)

#     for i in range(0, len(samples)):
#         mol_graph = samples[i][0]

#         ####################################################
#         # 1
#         self_feats[i, 0] = mol_graph.MolLogP
#         self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
#         self_feats[i, 2] = mol_graph.MaxEStateIndex
#         self_feats[i, 3] = mol_graph.SMR_VSA10
#         self_feats[i, 4] = mol_graph.Kappa2
#         ####################################################
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        

# def collate_emodel_elastic_7(samples):
#     self_feats = np.empty((len(samples), 7), dtype=np.float32)

#     for i in range(0, len(samples)):
#         mol_graph = samples[i][0]

#         ####################################################
#         # 1
#         self_feats[i, 0] = mol_graph.MolLogP
#         self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
#         self_feats[i, 2] = mol_graph.MaxEStateIndex
#         self_feats[i, 3] = mol_graph.SMR_VSA10
#         self_feats[i, 4] = mol_graph.Kappa2
#         # 6
#         self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
#         self_feats[i, 6] = mol_graph.PEOE_VSA13
#         ####################################################

#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# def collate_emodel_elastic_10(samples):
#     self_feats = np.empty((len(samples), 10), dtype=np.float32)

#     for i in range(0, len(samples)):
#         mol_graph = samples[i][0]

#         ####################################################
#         # 1
#         self_feats[i, 0] = mol_graph.MolLogP
#         self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
#         self_feats[i, 2] = mol_graph.MaxEStateIndex
#         self_feats[i, 3] = mol_graph.SMR_VSA10
#         self_feats[i, 4] = mol_graph.Kappa2
#         # 6
#         self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
#         self_feats[i, 6] = mol_graph.PEOE_VSA13
#         self_feats[i, 7] = mol_graph.MinAbsPartialCharge
#         self_feats[i, 8] = mol_graph.BCUT2D_CHGHI
#         self_feats[i, 9] = mol_graph.PEOE_VSA6
#         ####################################################

#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


# def collate_emodel_elastic_20(samples):
#     self_feats = np.empty((len(samples), 20), dtype=np.float32)

#     for i in range(0, len(samples)):
#         mol_graph = samples[i][0]

#         ####################################################
#         # 1
#         self_feats[i, 0] = mol_graph.MolLogP
#         self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
#         self_feats[i, 2] = mol_graph.MaxEStateIndex
#         self_feats[i, 3] = mol_graph.SMR_VSA10
#         self_feats[i, 4] = mol_graph.Kappa2
#         # 6
#         self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
#         self_feats[i, 6] = mol_graph.PEOE_VSA13
#         self_feats[i, 7] = mol_graph.MinAbsPartialCharge
#         self_feats[i, 8] = mol_graph.BCUT2D_CHGHI
#         self_feats[i, 9] = mol_graph.PEOE_VSA6
#         # 11
#         self_feats[i, 10] = mol_graph.SlogP_VSA1
#         self_feats[i, 11] = mol_graph.fr_nitro
#         self_feats[i, 12] = mol_graph.BalabanJ
#         self_feats[i, 13] = mol_graph.SMR_VSA9
#         self_feats[i, 14] = mol_graph.fr_alkyl_halide
#         # 16
#         self_feats[i, 15] = mol_graph.fr_hdrzine
#         self_feats[i, 16] = mol_graph.PEOE_VSA8
#         self_feats[i, 17] = mol_graph.fr_Ar_NH
#         self_feats[i, 18] = mol_graph.fr_imidazole
#         self_feats[i, 19] = mol_graph.fr_Nhpyrrole
#         ####################################################

#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
# ########################################################################################################


def collate_emodel_elastic(samples):
    self_feats = np.empty((len(samples), mc.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 1
        self_feats[i, 0] = mol_graph.NumValenceElectrons
        self_feats[i, 1] = mol_graph.Chi0n
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
########################################################################################################


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)

# # ####################################################################################
# # scaling이 필요할 때
# y_train = np.array([target for (_, target) in dataset])
# scaler = StandardScaler()
# y_train_scaling = scaler.fit_transform(y_train)
# # 5. 스케일된 target을 다시 덮어씌우기
# dataset = [(g, y) for (g, _), y in zip(dataset, y_train_scaling)]
# # ####################################################################################


####################################################################################
# ##### scaling part - train test 분리할때만 #####
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = SEED)

# scaling
y_train = np.array([target for (_, target) in train_dataset])
y_test = np.array([target for (_, target) in test_dataset])

scaler = StandardScaler()
y_train_scaling = scaler.fit_transform(y_train)
y_test_scaling = scaler.transform(y_test)

# 5. 스케일된 target을 다시 덮어씌우기
train_dataset = [(g, y) for (g, _), y in zip(train_dataset, y_train_scaling)]
test_dataset = [(g, y) for (g, _), y in zip(test_dataset, y_test_scaling)]
##### scaling part - train test 분리할때만 #####
####################################################################################


#=====================================================================#
#=========================== Embedding : 1 ===========================#
#=====================================================================#

# EGCN
model_EGCN_1 = EGCN_1_copy.Net(mc.dim_atomic_feat, 1, 1).to(device)
model_EGCN_3 = EGCN_3.Net(mc.dim_atomic_feat, 1, 3).to(device)
# model_EGCN_5 = EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
# model_EGCN_7 = EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
# model_EGCN_10 = EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
# model_EGCN_20 = EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Outer_EGCN
model_Outer_EGCN_3 = Outer_EGCN_3_copy.Net(mc.dim_atomic_feat, 1, 3).to(device)
# model_Outer_EGCN_5 = Outer_EGCN_5.Net(mc.dim_atomic_feat, 1, 5).to(device)
# model_Outer_EGCN_7 = Outer_EGCN_7.Net(mc.dim_atomic_feat, 1, 7).to(device)
# model_Outer_EGCN_10 = Outer_EGCN_10.Net(mc.dim_atomic_feat, 1, 10).to(device)
# model_Outer_EGCN_20 = Outer_EGCN_20.Net(mc.dim_atomic_feat, 1, 20).to(device)

# Self_Feature
model_EGCN_elastic = EGCN_elastic.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)
model_Outer_EGCN_elastic = Outer_EGCN_elastic_copy2.Net(mc.dim_atomic_feat, 1, mc.dim_self_feat).to(device)


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

# # feature 20개
# print('--------- EGCN_elastic ---------')
# test_losses['EGCN_elastic'] = trainer.cross_validation(dataset, model_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
# print('test loss (EGCN_elastic): ' + str(test_losses['EGCN_elastic']))

# #------------------------ Outer EGCN ------------------------#

# # feature 3개
# print('--------- Outer EGCN_3 ---------')
# test_losses['Outer_EGCN_3'] = trainer.cross_validation(dataset, model_Outer_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_3)
# print('test loss (Outer_EGCN_3): ' + str(test_losses['Outer_EGCN_3']))

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
# test_losses['Outer_EGCN_20'] = trainer.cross_validation(dataset, model_Outer_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic_20)
# print('test loss (Outer_EGCN_20): ' + str(test_losses['Outer_EGCN_20']))


# #------------------------ Self Feature ------------------------#

# print('--------- EGCN_elastic ---------')
# test_losses['EGCN_elastic'] = trainer.cross_validation(dataset, model_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
# print('test loss (EGCN_elastic): ' + str(test_losses['EGCN_elastic']))

# print('--------- Outer EGCN_elastic ---------')
# test_losses['Outer_EGCN_elastic'] = trainer.cross_validation(dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_elastic)
# print('test loss (Outer_EGCN_elastic): ' + str(test_losses['Outer_EGCN_elastic']))


#=====================================================================#
#=========================== Embedding : 2 ===========================#
#=====================================================================#




# 최종 평가
print('--------- Outer EGCN_elastic ---------')
val_losses['Outer_EGCN_elastic'], best_model, best_k = trainer_test.cross_validation(train_dataset, model_Outer_EGCN_elastic, criterion, k, batch_size, max_epochs, trainer_test.train_model, trainer_test.val_model, collate_emodel_elastic)
print('Val loss (Outer_EGCN_elastic): ' + str(val_losses['Outer_EGCN_elastic']))

test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_emodel_elastic)
test_loss, final_preds = trainer_test.test_model(best_model, criterion, test_data_loader)
print('best_k-fold:', best_k)
print(val_losses)
print(test_loss)
# =====================================================================#
# =========================== Embedding : 2 ===========================#
# =====================================================================#

print('best_k-fold:', best_k)
print(test_losses)
