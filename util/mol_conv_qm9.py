import pandas
import torch
import dgl
import numpy as np
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
#from mendeleev import get_table
from mendeleev.fetch import fetch_table

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
dim_self_feat = 20


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
    molGraph = molDGLGraph(smiles, adj_mat, feat_mat, mol)
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
    except:
        print(smiles + ' could not be converted to molecular graph due to the internal errors of RDKit')
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
            # qm9_reduced
            # 3
            mol_graph.HeavyAtomCount = dsc.HeavyAtomCount(mol)
            mol_graph.Chi0 = dsc.Chi0(mol)
            mol_graph.Chi0v = dsc.Chi0v(mol)
            # 5
            mol_graph.TPSA = dsc.TPSA(mol)
            mol_graph.fr_Al_OH = dsc.fr_Al_OH(mol)
            # 7
            mol_graph.NumHAcceptors = dsc.NumHAcceptors(mol)
            mol_graph.Chi4v = dsc.Chi4v(mol)
            # 10
            mol_graph.NumRotatableBonds = dsc.NumRotatableBonds(mol)
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            # 20
            mol_graph.BertzCT = dsc.BertzCT(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.Kappa1 = dsc.Kappa1(mol)
            mol_graph.NumSaturatedCarbocycles = dsc.NumSaturatedCarbocycles(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.VSA_EState4 = dsc.VSA_EState4(mol)
            mol_graph.VSA_EState7 = dsc.VSA_EState7(mol)
            mol_graph.fr_Ar_OH = dsc.fr_Ar_OH(mol)

            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)


    ####################################################
    # qm9_reduced
    # 3
    normalize_self_feat(mol_graphs, 'HeavyAtomCount')
    normalize_self_feat(mol_graphs, 'Chi0')
    normalize_self_feat(mol_graphs, 'Chi0v')
    # 5
    normalize_self_feat(mol_graphs, 'TPSA')
    normalize_self_feat(mol_graphs, 'fr_Al_OH')
    # 7
    normalize_self_feat(mol_graphs, 'NumHAcceptors')
    normalize_self_feat(mol_graphs, 'Chi4v')
    # 10
    normalize_self_feat(mol_graphs, 'NumRotatableBonds')
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGHI')
    normalize_self_feat(mol_graphs, 'SMR_VSA9')
    # 20
    normalize_self_feat(mol_graphs, 'BertzCT')
    normalize_self_feat(mol_graphs, 'PEOE_VSA4')
    normalize_self_feat(mol_graphs, 'Kappa1')
    normalize_self_feat(mol_graphs, 'NumSaturatedCarbocycles')
    normalize_self_feat(mol_graphs, 'MolLogP')
    normalize_self_feat(mol_graphs, 'BalabanJ')
    normalize_self_feat(mol_graphs, 'SlogP_VSA10')
    normalize_self_feat(mol_graphs, 'VSA_EState4')
    normalize_self_feat(mol_graphs, 'VSA_EState7')
    normalize_self_feat(mol_graphs, 'fr_Ar_OH')
    ####################################################

    return samples

atomic_props = read_atom_prop()