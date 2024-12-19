import pandas
import torch
import dgl
import numpy as np
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
#from mendeleev import get_table
from mendeleev.fetch import fetch_table
import traceback # 예외정보 추적

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sel_prop_names = ['atomic_weight',
                'atomic_radius',
                'atomic_volume',
                'dipole_polarizability',
                'fusion_heat',
                'thermal_conductivity',
                'vdw_radius',
                'en_pauling']
dim_atomic_feat = len(sel_prop_names)
dim_self_feat = 37


class molDGLGraph(dgl.DGLGraph):
    def __init__(self, smiles, adj_mat, feat_mat, mol):
        super(molDGLGraph, self).__init__()
        self.smiles = smiles
        self.adj_mat = adj_mat
        self.feat_mat = feat_mat
        self.atomic_nodes = []
        self.neighbors = {}

        node_id = 0
        for atom in mol.GetAtoms():
            self.atomic_nodes.append(atom.GetSymbol())
            self.neighbors[node_id] = atoms_to_symbols(atom.GetNeighbors())
            node_id += 1


def read_atom_prop():
#    tb_atomic_props = get_table('elements')
    tb_atomic_props = fetch_table('elements')
#    arr_atomic_nums = np.array(tb_atomic_props['atomic_number'], dtype=np.int)
    arr_atomic_nums = np.array(tb_atomic_props['atomic_number'], dtype=int)
#    arr_atomic_props = np.nan_to_num(np.array(tb_atomic_props[sel_prop_names], dtype=np.float32))
    arr_atomic_props = np.nan_to_num(np.array(tb_atomic_props[sel_prop_names], dtype=float))
    arr_atomic_props = util.zscore(arr_atomic_props)
    atomic_props_mat = {arr_atomic_nums[i]: arr_atomic_props[i, :] for i in range(0, arr_atomic_nums.shape[0])}

    return atomic_props_mat


def construct_mol_graph(smiles, mol, adj_mat, feat_mat):
    molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol).to(device)
    edges = util.adj_mat_to_edges(adj_mat)
    src, dst = tuple(zip(*edges))

    molGraph.add_nodes(adj_mat.shape[0])
    molGraph.add_edges(src, dst)
    molGraph.ndata['feat'] = torch.tensor(feat_mat, dtype=torch.float32).to(device)

    return molGraph


def smiles_to_mol_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        node_feat_mat = np.empty([mol.GetNumAtoms(), atomic_props.get(1).shape[0]])

        ind = 0
        for atom in mol.GetAtoms():
            node_feat_mat[ind, :] = atomic_props.get(atom.GetAtomicNum())
            ind = ind + 1

        return mol, construct_mol_graph(smiles, mol, adj_mat, node_feat_mat)
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(traceback.format_exc())  # 예외 정보를 출력
        return None, None


def atoms_to_symbols(atoms):
    symbols = []

    for atom in atoms:
        symbols.append(atom.GetSymbol())

    return symbols


def normalize_self_feat(mol_graphs, self_feat_name):
    self_feats = []

    for mol_graph in mol_graphs:
        self_feats.append(getattr(mol_graph, self_feat_name))

    mean_self_feat = np.mean(self_feats)
    std_self_feat = np.std(self_feats)

    for mol_graph in mol_graphs:
        if std_self_feat == 0:
            setattr(mol_graph, self_feat_name, 0)
        else:
            setattr(mol_graph, self_feat_name, (getattr(mol_graph, self_feat_name) - mean_self_feat) / std_self_feat)

"""
Dataset: lipo
"""

def read_dataset(file_name):
    samples = []
    mol_graphs = []
    data_mat = np.array(pandas.read_csv(file_name))
    smiles = data_mat[:, 0]
#    target = np.array(data_mat[:, 1:3], dtype=np.float)
    target = np.array(data_mat[:, 1:3], dtype=float)

    for i in range(0, data_mat.shape[0]):
        mol, mol_graph = smiles_to_mol_graph(smiles[i])

        if mol is not None and mol_graph is not None:

            ####################################################
            # 1
            mol_graph.fr_COO = dsc.fr_COO(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            mol_graph.SMR_VSA1 = dsc.SMR_VSA1(mol)
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            # 6
            mol_graph.NumHAcceptors = dsc.NumHAcceptors(mol)
            mol_graph.fr_NH1 = dsc.fr_NH1(mol)
            mol_graph.FpDensityMorgan3 = dsc.FpDensityMorgan3(mol)
            mol_graph.Chi4v = dsc.Chi4v(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            # 11
            mol_graph.PEOE_VSA6 = dsc.PEOE_VSA6(mol)
            mol_graph.VSA_EState5 = dsc.VSA_EState5(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            # 16
            mol_graph.NumSaturatedCarbocycles = dsc.NumSaturatedCarbocycles(mol)
            mol_graph.VSA_EState2 = dsc.VSA_EState2(mol)
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.qed = dsc.qed(mol)
            mol_graph.MinAbsEStateIndex = dsc.MinAbsEStateIndex(mol)
            # 21
            mol_graph.MinEStateIndex = dsc.MinEStateIndex(mol)
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            mol_graph.fr_unbrch_alkane = dsc.fr_unbrch_alkane(mol)
            mol_graph.fr_HOCCN = dsc.fr_HOCCN(mol)
            mol_graph.fr_sulfide = dsc.fr_sulfide(mol)
            # 26
            mol_graph.EState_VSA5 = dsc.EState_VSA5(mol)
            mol_graph.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(mol)
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.fr_phenol = dsc.fr_phenol(mol)
            mol_graph.fr_bicyclic = dsc.fr_bicyclic(mol)
            # 31
            mol_graph.PEOE_VSA12 = dsc.PEOE_VSA12(mol)
            mol_graph.fr_guanido = dsc.fr_guanido(mol)
            mol_graph.fr_lactone = dsc.fr_lactone(mol)
            mol_graph.fr_furan = dsc.fr_furan(mol)
            mol_graph.fr_piperzine = dsc.fr_piperzine(mol)
            # 36
            mol_graph.fr_NH2 = dsc.fr_NH2(mol)
            mol_graph.fr_amidine = dsc.fr_amidine(mol)
            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # 1
    normalize_self_feat(mol_graphs, 'fr_COO')
    normalize_self_feat(mol_graphs, 'MolLogP')
    normalize_self_feat(mol_graphs, 'NumAromaticHeterocycles')
    normalize_self_feat(mol_graphs, 'SMR_VSA1')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan1')
    # 6
    normalize_self_feat(mol_graphs, 'NumHAcceptors')
    normalize_self_feat(mol_graphs, 'fr_NH1')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan3')
    normalize_self_feat(mol_graphs, 'Chi4v')
    normalize_self_feat(mol_graphs, 'SMR_VSA10')
    # 11
    normalize_self_feat(mol_graphs, 'PEOE_VSA6')
    normalize_self_feat(mol_graphs, 'VSA_EState5')
    normalize_self_feat(mol_graphs, 'SlogP_VSA10')
    normalize_self_feat(mol_graphs, 'PEOE_VSA13')
    normalize_self_feat(mol_graphs, 'fr_pyridine')
    # 16
    normalize_self_feat(mol_graphs, 'NumSaturatedCarbocycles')
    normalize_self_feat(mol_graphs, 'VSA_EState2')
    normalize_self_feat(mol_graphs, 'MaxEStateIndex')
    normalize_self_feat(mol_graphs, 'qed')
    normalize_self_feat(mol_graphs, 'MinAbsEStateIndex')
    # 21
    normalize_self_feat(mol_graphs, 'MinEStateIndex')
    normalize_self_feat(mol_graphs, 'SlogP_VSA4')
    normalize_self_feat(mol_graphs, 'fr_unbrch_alkane')
    normalize_self_feat(mol_graphs, 'fr_HOCCN')
    normalize_self_feat(mol_graphs, 'fr_sulfide')
    # 26
    normalize_self_feat(mol_graphs, 'EState_VSA5')
    normalize_self_feat(mol_graphs, 'fr_nitro_arom_nonortho')
    normalize_self_feat(mol_graphs, 'Ipc')
    normalize_self_feat(mol_graphs, 'fr_phenol')
    normalize_self_feat(mol_graphs, 'fr_bicyclic')
    # 31
    normalize_self_feat(mol_graphs, 'PEOE_VSA12')
    normalize_self_feat(mol_graphs, 'fr_guanido')
    normalize_self_feat(mol_graphs, 'fr_lactone')
    normalize_self_feat(mol_graphs, 'fr_furan')
    normalize_self_feat(mol_graphs, 'fr_piperzine')
    # 36
    normalize_self_feat(mol_graphs, 'fr_NH2')
    normalize_self_feat(mol_graphs, 'fr_amidine')
    ####################################################

    return samples


atomic_props = read_atom_prop()