import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv_esol as mc
import util.mol_conv_esol_sf as mc_sf
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
from model import Extended_EGCN_sf

from model import Bilinear_EGCN
from model import test_EGCN
from util import trainer

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
dataset_name = 'esol'
batch_size = 32
max_epochs = 5
k = 2


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


"""
esol 용
"""
########################################################################################################
# esol
def collate_emodel_Extended_3(samples):
    self_feats = np.empty((len(samples), 3), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 3
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_5(samples):
    self_feats = np.empty((len(samples), 5), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 3
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        # 5
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
        

def collate_emodel_Extended_7(samples):
    self_feats = np.empty((len(samples), 7), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 3
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        # 5
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        # 7
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.fr_imide
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_10(samples):
    self_feats = np.empty((len(samples), 10), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 3
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        # 5
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        # 7
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.fr_imide
        # 10
        self_feats[i, 7] = mol_graph.Kappa2
        self_feats[i, 8] = mol_graph.MinAbsPartialCharge
        self_feats[i, 9] = mol_graph.NumAromaticHeterocycles
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)


def collate_emodel_Extended_20(samples):
    self_feats = np.empty((len(samples), 20), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]

        ####################################################
        # 3
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.SMR_VSA10
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        # 5
        self_feats[i, 3] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 4] = mol_graph.BCUT2D_CHGHI
        # 7
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.fr_imide
        # 10
        self_feats[i, 7] = mol_graph.Kappa2
        self_feats[i, 8] = mol_graph.MinAbsPartialCharge
        self_feats[i, 9] = mol_graph.NumAromaticHeterocycles
        # 20
        self_feats[i, 10] = mol_graph.SlogP_VSA1
        self_feats[i, 11] = mol_graph.fr_amide
        self_feats[i, 12] = mol_graph.BalabanJ
        self_feats[i, 13] = mol_graph.fr_Ar_NH
        self_feats[i, 14] = mol_graph.PEOE_VSA8
        self_feats[i, 15] = mol_graph.NumSaturatedRings
        self_feats[i, 16] = mol_graph.fr_NH0
        self_feats[i, 17] = mol_graph.PEOE_VSA13
        self_feats[i, 18] = mol_graph.fr_barbitur
        self_feats[i, 19] = mol_graph.fr_alkyl_halide
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
########################################################################################################


def collate_emodel_Extended_sf(samples):
    self_feats = np.empty((len(samples), mc_sf.dim_self_feat), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        ####################################################
        # 3
        self_feats[i, 0] = mol_graph.MaxEStateIndex
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MinAbsPartialCharge
        # 5
        self_feats[i, 3] = mol_graph.FpDensityMorgan2
        self_feats[i, 4] = mol_graph.BCUT2D_MWLOW
        # 7
        self_feats[i, 5] = mol_graph.BCUT2D_CHGHI
        self_feats[i, 6] = mol_graph.BalabanJ
        # 10
        self_feats[i, 7] = mol_graph.Kappa2
        self_feats[i, 8] = mol_graph.PEOE_VSA13
        self_feats[i, 9] = mol_graph.PEOE_VSA8
        # 11
        self_feats[i, 10] = mol_graph.PEOE_VSA9
        self_feats[i, 11] = mol_graph.SMR_VSA10
        self_feats[i, 12] = mol_graph.SMR_VSA9
        self_feats[i, 13] = mol_graph.SlogP_VSA1
        self_feats[i, 14] = mol_graph.SlogP_VSA10
        # 16
        self_feats[i, 15] = mol_graph.SlogP_VSA7
        self_feats[i, 16] = mol_graph.EState_VSA10
        self_feats[i, 17] = mol_graph.EState_VSA11
        self_feats[i, 18] = mol_graph.EState_VSA2
        self_feats[i, 19] = mol_graph.EState_VSA5
        # 21
        self_feats[i, 20] = mol_graph.EState_VSA8
        self_feats[i, 21] = mol_graph.FractionCSP3
        self_feats[i, 22] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 33] = mol_graph.NumSaturatedRings
        self_feats[i, 24] = mol_graph.MolLogP
        # 26
        self_feats[i, 25] = mol_graph.fr_Al_OH_noTert
        self_feats[i, 26] = mol_graph.fr_Ar_NH
        self_feats[i, 27] = mol_graph.fr_NH0
        self_feats[i, 28] = mol_graph.fr_alkyl_halide
        self_feats[i, 29] = mol_graph.fr_amide
        # 31
        self_feats[i, 30] = mol_graph.fr_aryl_methyl
        self_feats[i, 31] = mol_graph.fr_barbitur
        self_feats[i, 32] = mol_graph.fr_bicyclic
        self_feats[i, 33] = mol_graph.fr_ester
        self_feats[i, 34] = mol_graph.fr_furan
        # 36
        self_feats[i, 35] = mol_graph.fr_hdrzine
        self_feats[i, 36] = mol_graph.fr_imidazole
        self_feats[i, 37] = mol_graph.fr_imide
        self_feats[i, 38] = mol_graph.fr_nitro_arom_nonortho
        self_feats[i, 39] = mol_graph.fr_para_hydroxylation
        # 41
        self_feats[i, 40] = mol_graph.fr_phos_acid
        ####################################################

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).to(device)
########################################################################################################


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
dataset_sf = mc_sf.read_dataset('data/' + dataset_name + '.csv')

random.shuffle(dataset)
random.shuffle(dataset_sf)


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

model_Extended_EGCN_sf = Extended_EGCN_sf.Net(mc_sf.dim_atomic_feat, 1, mc_sf.dim_self_feat).to(device)

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


#=====================================================================#

# # feature 3개
# print('--------- EGCN_3 ---------')
# test_losses['EGCN_3'] = trainer.cross_validation(dataset, model_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_3)
# print('test loss (EGCN_3): ' + str(test_losses['EGCN_3']))

# # feature 5개
# print('--------- EGCN_5 ---------')
# test_losses['EGCN_5'] = trainer.cross_validation(dataset, model_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_5)
# print('test loss (EGCN_5): ' + str(test_losses['EGCN_5']))

# # feature 7개
# print('--------- EGCN_7 ---------')
# test_losses['EGCN_7'] = trainer.cross_validation(dataset, model_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_7)
# print('test loss (EGCN_7): ' + str(test_losses['EGCN_7']))

# # feature 10개
# print('--------- EGCN_10 ---------')
# test_losses['EGCN_10'] = trainer.cross_validation(dataset, model_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_10)
# print('test loss (EGCN_10): ' + str(test_losses['EGCN_10']))

# # feature 20개
# print('--------- EGCN_20 ---------')
# test_losses['EGCN_20'] = trainer.cross_validation(dataset, model_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_20)
# print('test loss (EGCN_20): ' + str(test_losses['EGCN_20']))


#=====================================================================#

# Extended_EGCN
# # feature 3개
# print('--------- Exteded EGCN_3 ---------')
# test_losses['Extended_EGCN_3'] = trainer.cross_validation(dataset, model_Extended_EGCN_3, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_3)
# print('test loss (Extended_EGCN_3): ' + str(test_losses['Extended_EGCN_3']))

# # feature 5개
# print('--------- Exteded EGCN_5 ---------')
# test_losses['Extended_EGCN_5'] = trainer.cross_validation(dataset, model_Extended_EGCN_5, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_5)
# print('test loss (Extended_EGCN_5): ' + str(test_losses['Extended_EGCN_5']))

# # feature 7개
# print('--------- Exteded EGCN_7 ---------')
# test_losses['Extended_EGCN_7'] = trainer.cross_validation(dataset, model_Extended_EGCN_7, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_7)
# print('test loss (Extended_EGCN_7): ' + str(test_losses['Extended_EGCN_7']))

# # feature 10개
# print('--------- Exteded EGCN_10 ---------')
# test_losses['Extended_EGCN_10'] = trainer.cross_validation(dataset, model_Extended_EGCN_10, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_10)
# print('test loss (Extended_EGCN_10): ' + str(test_losses['Extended_EGCN_10']))

# # feature 20개
# print('--------- Exteded EGCN_20 ---------')
# test_losses['Extended_EGCN_20'] = trainer.cross_validation(dataset, model_Extended_EGCN_20, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_20)
# print('test loss (Extended_EGCN_20): ' + str(test_losses['Extended_EGCN_20']))


#=====================================================================#

# self_feature
print('--------- Exteded EGCN_sf ---------')
test_losses['Extended_EGCN_sf'] = trainer.cross_validation(dataset_sf, model_Extended_EGCN_sf, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_Extended_sf)
print('test loss (Extended_EGCN_sf): ' + str(test_losses['Extended_EGCN_sf']))


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
