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
            # qm7_reduced
            # 3
            mol_graph.Chi1n = dsc.Chi1n(mol)
            mol_graph.Chi1v = dsc.Chi1v(mol)
            mol_graph.Chi0v = dsc.Chi0v(mol)
            # 5
            mol_graph.MaxPartialCharge = dsc.MaxPartialCharge(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            # 7
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            mol_graph.EState_VSA9 = dsc.EState_VSA9(mol)
            # 10
            mol_graph.VSA_EState2 = dsc.VSA_EState2(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.SlogP_VSA3 = dsc.SlogP_VSA3(mol)
            # 20
            mol_graph.BCUT2D_MRLOW = dsc.BCUT2D_MRLOW(mol)
            mol_graph.PEOE_VSA11 = dsc.PEOE_VSA11(mol)
            mol_graph.SlogP_VSA8 = dsc.SlogP_VSA8(mol)
            mol_graph.EState_VSA7 = dsc.EState_VSA7(mol)
            mol_graph.fr_Ndealkylation1 = dsc.fr_Ndealkylation1(mol)
            mol_graph.SMR_VSA5 = dsc.SMR_VSA5(mol)
            mol_graph.fr_epoxide = dsc.fr_epoxide(mol)
            mol_graph.fr_NH1 = dsc.fr_NH1(mol)
            mol_graph.VSA_EState7 = dsc.VSA_EState7(mol)
            mol_graph.qed = dsc.qed(mol)
            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)


    ####################################################
    # qm7_reduced
    # 3
    normalize_self_feat(mol_graphs, 'Chi1n')
    normalize_self_feat(mol_graphs, 'Chi1v')
    normalize_self_feat(mol_graphs, 'Chi0v')
    # 5
    normalize_self_feat(mol_graphs, 'MaxPartialCharge')
    normalize_self_feat(mol_graphs, 'Kappa2')
    # 7
    normalize_self_feat(mol_graphs, 'SlogP_VSA4')
    normalize_self_feat(mol_graphs, 'EState_VSA9')
    # 10
    normalize_self_feat(mol_graphs, 'VSA_EState2')
    normalize_self_feat(mol_graphs, 'PEOE_VSA4')
    normalize_self_feat(mol_graphs, 'SlogP_VSA3')
    # 20
    normalize_self_feat(mol_graphs, 'BCUT2D_MRLOW')
    normalize_self_feat(mol_graphs, 'PEOE_VSA11')
    normalize_self_feat(mol_graphs, 'SlogP_VSA8')
    normalize_self_feat(mol_graphs, 'EState_VSA7')
    normalize_self_feat(mol_graphs, 'fr_Ndealkylation1')
    normalize_self_feat(mol_graphs, 'SMR_VSA5')
    normalize_self_feat(mol_graphs, 'fr_epoxide')
    normalize_self_feat(mol_graphs, 'fr_NH1')
    normalize_self_feat(mol_graphs, 'VSA_EState7')
    normalize_self_feat(mol_graphs, 'qed')
    ####################################################

    return samples

atomic_props = read_atom_prop()