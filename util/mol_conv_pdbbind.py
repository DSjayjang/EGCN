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
dim_self_feat = 38


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
            mol_graph.VSA_EState6 = dsc.VSA_EState6(mol)
            mol_graph.Chi4n = dsc.Chi4n(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            mol_graph.fr_halogen = dsc.fr_halogen(mol)
            # 6
            mol_graph.FractionCSP3 = dsc.FractionCSP3(mol)
            mol_graph.SlogP_VSA6 = dsc.SlogP_VSA6(mol)
            mol_graph.PEOE_VSA3 = dsc.PEOE_VSA3(mol)
            mol_graph.SlogP_VSA3 = dsc.SlogP_VSA3(mol)
            mol_graph.fr_Ndealkylation2 = dsc.fr_Ndealkylation2(mol)
            # 11
            mol_graph.fr_Al_COO = dsc.fr_Al_COO(mol)
            mol_graph.fr_NH2 = dsc.fr_NH2(mol)
            mol_graph.MinAbsEStateIndex = dsc.MinAbsEStateIndex(mol)
            mol_graph.PEOE_VSA9 = dsc.PEOE_VSA9(mol)
            mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            # 16
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.fr_ArN = dsc.fr_ArN(mol)
            mol_graph.fr_imidazole = dsc.fr_imidazole(mol)
            mol_graph.fr_allylic_oxid = dsc.fr_allylic_oxid(mol)
            # 21
            mol_graph.EState_VSA9 = dsc.EState_VSA9(mol)
            mol_graph.VSA_EState4 = dsc.VSA_EState4(mol)
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            mol_graph.fr_sulfonamd = dsc.fr_sulfonamd(mol)
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            # 26
            mol_graph.EState_VSA4 = dsc.EState_VSA4(mol)
            mol_graph.fr_alkyl_carbamate = dsc.fr_alkyl_carbamate(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.fr_N_O = dsc.fr_N_O(mol)
            mol_graph.fr_unbrch_alkane = dsc.fr_unbrch_alkane(mol)
            # 31
            mol_graph.fr_Ar_COO = dsc.fr_Ar_COO(mol)
            mol_graph.fr_nitrile = dsc.fr_nitrile(mol)
            mol_graph.fr_sulfone = dsc.fr_sulfone(mol)
            mol_graph.VSA_EState8 = dsc.VSA_EState8(mol)
            mol_graph.fr_thiazole = dsc.fr_thiazole(mol)
            # 36
            mol_graph.VSA_EState9 = dsc.VSA_EState9(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # 1
    normalize_self_feat(mol_graphs, 'VSA_EState6')
    normalize_self_feat(mol_graphs, 'Chi4n')
    normalize_self_feat(mol_graphs, 'MolLogP')
    normalize_self_feat(mol_graphs, 'NumAromaticHeterocycles')
    normalize_self_feat(mol_graphs, 'fr_halogen')
    # 6
    normalize_self_feat(mol_graphs, 'FractionCSP3')
    normalize_self_feat(mol_graphs, 'SlogP_VSA6')
    normalize_self_feat(mol_graphs, 'PEOE_VSA3')
    normalize_self_feat(mol_graphs, 'SlogP_VSA3')
    normalize_self_feat(mol_graphs, 'fr_Ndealkylation2')
    # 11
    normalize_self_feat(mol_graphs, 'fr_Al_COO')
    normalize_self_feat(mol_graphs, 'fr_NH2')
    normalize_self_feat(mol_graphs, 'MinAbsEStateIndex')
    normalize_self_feat(mol_graphs, 'PEOE_VSA9')
    normalize_self_feat(mol_graphs, 'fr_alkyl_halide')
    # 16
    normalize_self_feat(mol_graphs, 'MaxEStateIndex')
    normalize_self_feat(mol_graphs, 'BalabanJ')
    normalize_self_feat(mol_graphs, 'fr_ArN')
    normalize_self_feat(mol_graphs, 'fr_imidazole')
    normalize_self_feat(mol_graphs, 'fr_allylic_oxid')
    # 21
    normalize_self_feat(mol_graphs, 'EState_VSA9')
    normalize_self_feat(mol_graphs, 'VSA_EState4')
    normalize_self_feat(mol_graphs, 'PEOE_VSA13')
    normalize_self_feat(mol_graphs, 'fr_sulfonamd')
    normalize_self_feat(mol_graphs, 'SMR_VSA9')
    # 26
    normalize_self_feat(mol_graphs, 'EState_VSA4')
    normalize_self_feat(mol_graphs, 'fr_alkyl_carbamate')
    normalize_self_feat(mol_graphs, 'PEOE_VSA4')
    normalize_self_feat(mol_graphs, 'fr_N_O')
    normalize_self_feat(mol_graphs, 'fr_unbrch_alkane')
    # 31
    normalize_self_feat(mol_graphs, 'fr_Ar_COO')
    normalize_self_feat(mol_graphs, 'fr_nitrile')
    normalize_self_feat(mol_graphs, 'fr_sulfone')
    normalize_self_feat(mol_graphs, 'VSA_EState8')
    normalize_self_feat(mol_graphs, 'fr_thiazole')
    # 36
    normalize_self_feat(mol_graphs, 'VSA_EState9')
    normalize_self_feat(mol_graphs, 'fr_Ar_NH')
    normalize_self_feat(mol_graphs, 'fr_pyridine')
    ####################################################

    return samples


atomic_props = read_atom_prop()