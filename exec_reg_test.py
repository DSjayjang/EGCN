import torch
import torch.nn as nn
import dgl
import random
import numpy as np
import util.mol_conv as mc
from model import GCN
from model import GAT
from model import EGCN
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
dataset_name = 'molproperty_mp'
batch_size = 32
max_epochs = 300
k = 5


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

########################################################################################################
# default
# don't touch
# ring 1개
def collate_emodel_ring(samples):
    self_feats = np.empty((len(samples), 1), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_rings

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)
    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

# num_atoms + weight 2개
def collate_emodel_scale(samples):
    self_feats = np.empty((len(samples), 2), dtype=np.float32)

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        self_feats[i, 0] = mol_graph.num_atoms
        self_feats[i, 1] = mol_graph.weight

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    # return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels).view(-1, 1).to(device)
    return batched_graph, torch.tensor(self_feats).to(device), torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)


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
########################################################################################################


# load train, validation, and test datasets
print('Data loading...')
dataset = mc.read_dataset('data/' + dataset_name + '.csv')
random.shuffle(dataset)
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = SEED)
####################################################################################
# scaling이 필요할 때
y_train = np.array([target for (_, target) in dataset])
scaler = StandardScaler()
y_train_scaling = scaler.fit_transform(y_train)
# 5. 스케일된 target을 다시 덮어씌우기
dataset = [(g, y) for (g, _), y in zip(dataset, y_train_scaling)]
####################################################################################



#=====================================================================#
# default model
# don't touch
model_GCN = GCN.Net(mc.dim_atomic_feat, 1).to(device)
model_GAT = GAT.Net(mc.dim_atomic_feat, 1, 4).to(device)
model_EGCN_R = EGCN.Net(mc.dim_atomic_feat, 1, 1).to(device)
model_EGCN_S = EGCN.Net(mc.dim_atomic_feat, 1, 2).to(device)
model_EGCN = EGCN.Net(mc.dim_atomic_feat, 1, 3).to(device)
#=====================================================================#


# define loss function
criterion = nn.L1Loss(reduction='sum')
# criterion = nn.MSELoss(reduction='sum')

# train and evaluate competitors
test_losses = dict()


#=====================================================================#
# default model
# don't touch

print('--------- GCN ---------')
test_losses['GCN'] = trainer.cross_validation(dataset, model_GCN, criterion, k, batch_size, max_epochs, trainer.train, trainer.test, collate)
print('test loss (GCN): ' + str(test_losses['GCN']))

print('--------- GAT ---------')
test_losses['GAT'] = trainer.cross_validation(dataset, model_GAT, criterion, k, batch_size, max_epochs, trainer.train, trainer.test, collate)
print('test loss (GAT): ' + str(test_losses['GAT']))

print('--------- EGCN_RING ---------')
test_losses['EGCN_R'] = trainer.cross_validation(dataset, model_EGCN_R, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_ring)
print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

print('--------- EGCN_SCALE ---------')
test_losses['EGCN_S'] = trainer.cross_validation(dataset, model_EGCN_S, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel_scale)
print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

print('--------- EGCN ---------')
test_losses['EGCN'] = trainer.cross_validation(dataset, model_EGCN, criterion, k, batch_size, max_epochs, trainer.train_emodel, trainer.test_emodel, collate_emodel)
print('test loss (EGCN): ' + str(test_losses['EGCN']))

print(test_losses)

# #=====================================================================#

# print('--------- GCN ---------')
# test_losses['GCN'] = trainer_test.cross_validation(dataset, model_GCN, criterion, k, batch_size, max_epochs, trainer_test.train, trainer_test.test, collate)
# print('test loss (GCN): ' + str(test_losses['GCN']))

# print('--------- EGCN_RING ---------')
# test_losses['EGCN_R'] = trainer_test.cross_validation(dataset, model_EGCN_R, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel_ring)
# print('test loss (EGCN_RING): ' + str(test_losses['EGCN_R']))

# print('--------- EGCN_SCALE ---------')
# test_losses['EGCN_S'] = trainer_test.cross_validation(dataset, model_EGCN_S, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel_scale)
# print('test loss (EGCN_SCALE): ' + str(test_losses['EGCN_S']))

# print('--------- EGCN ---------')
# test_losses['EGCN'], best_model = trainer_test.cross_validation(dataset, model_EGCN, criterion, k, batch_size, max_epochs, trainer_test.train_emodel, trainer_test.test_emodel, collate_emodel)
# print('test loss (EGCN): ' + str(test_losses['EGCN']))



# print('--------- EGCN ---------')
# test_losses['EGCN'], best_model, best_k = trainer_test.cross_validation(dataset, model_EGCN, criterion, k, batch_size, max_epochs, trainer_test.train_model, trainer_test.val_model, collate_emodel)
# print('test loss (EGCN): ' + str(test_losses['EGCN']))
# print(test_losses)

# # 최종 평가
# """
# need to split the dataset to train and test dataset
# """
# test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_emodel)
# final_test_loss, final_preds = trainer_test.test_model(best_model, criterion, test_data_loader)

#=====================================================================#
#=========================== Embedding : 2 ===========================#
#=====================================================================#

# print('best_k-fold:', best_k)
# print(test_losses)
