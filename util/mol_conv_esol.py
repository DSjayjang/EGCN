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
            # esol
            # 3
            mol_graph.MinPartialCharge = dsc.MinPartialCharge(mol)
            mol_graph.SlogP_VSA2 = dsc.SlogP_VSA2(mol)
            mol_graph.MolMR = dsc.MolMR(mol)
            # 5
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            mol_graph.SlogP_VSA6 = dsc.SlogP_VSA6(mol)
            # 7
            mol_graph.SMR_VSA5 = dsc.SMR_VSA5(mol)
            mol_graph.HeavyAtomCount = dsc.HeavyAtomCount(mol)
            # 10
            mol_graph.FpDensityMorgan3 = dsc.FpDensityMorgan3(mol)
            mol_graph.NumHAcceptors = dsc.NumHAcceptors(mol)
            mol_graph.RingCount = dsc.RingCount(mol)
            # 20
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            mol_graph.EState_VSA9 = dsc.EState_VSA9(mol)
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            mol_graph.MinEStateIndex = dsc.MinEStateIndex(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            mol_graph.SlogP_VSA5 = dsc.SlogP_VSA5(mol)
            mol_graph.SlogP_VSA8 = dsc.SlogP_VSA8(mol)
            mol_graph.VSA_EState7 = dsc.VSA_EState7(mol)
            mol_graph.fr_C_O_noCOO = dsc.fr_C_O_noCOO(mol)
            ####################################################

            # ####################################################
            # # esol
            # # 3
            # mol_graph.MolLogP = dsc.MolLogP(mol)
            # mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            # mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            # # 5
            # mol_graph.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(mol)
            # mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            # # 7
            # mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            # mol_graph.fr_imide = dsc.fr_imide(mol)
            # # 10
            # mol_graph.Kappa2 = dsc.Kappa2(mol)
            # mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            # mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            # # 20
            # mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            # mol_graph.fr_amide = dsc.fr_amide(mol)
            # mol_graph.BalabanJ = dsc.BalabanJ(mol)
            # mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            # mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            # mol_graph.NumSaturatedRings = dsc.NumSaturatedRings(mol)
            # mol_graph.fr_NH0 = dsc.fr_NH0(mol)
            # mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            # mol_graph.fr_barbitur = dsc.fr_barbitur(mol)
            # mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            # ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # esol
    # 3
    normalize_self_feat(mol_graphs, 'MinPartialCharge')
    normalize_self_feat(mol_graphs, 'SlogP_VSA2')
    normalize_self_feat(mol_graphs, 'MolMR')
    # 5
    normalize_self_feat(mol_graphs, 'FpDensityMorgan1')
    normalize_self_feat(mol_graphs, 'SlogP_VSA6')
    # 7
    normalize_self_feat(mol_graphs, 'SMR_VSA5')
    normalize_self_feat(mol_graphs, 'HeavyAtomCount')
    # 10
    normalize_self_feat(mol_graphs, 'FpDensityMorgan3')
    normalize_self_feat(mol_graphs, 'NumHAcceptors')
    normalize_self_feat(mol_graphs, 'RingCount')
    # 20
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGHI')
    normalize_self_feat(mol_graphs, 'EState_VSA9')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan2')
    normalize_self_feat(mol_graphs, 'MinAbsPartialCharge')
    normalize_self_feat(mol_graphs, 'MinEStateIndex')
    normalize_self_feat(mol_graphs, 'NumAromaticHeterocycles')
    normalize_self_feat(mol_graphs, 'SlogP_VSA5')
    normalize_self_feat(mol_graphs, 'SlogP_VSA8')
    normalize_self_feat(mol_graphs, 'VSA_EState7')
    normalize_self_feat(mol_graphs, 'fr_C_O_noCOO')
    ####################################################

    # ####################################################
    # # esol
    # # 3
    # normalize_self_feat(mol_graphs, 'MolLogP')
    # normalize_self_feat(mol_graphs, 'SMR_VSA10')
    # normalize_self_feat(mol_graphs, 'MaxEStateIndex')
    # # 5
    # normalize_self_feat(mol_graphs, 'MaxAbsPartialCharge')
    # normalize_self_feat(mol_graphs, 'BCUT2D_CHGHI')
    # # 7
    # normalize_self_feat(mol_graphs, 'BCUT2D_MWLOW')
    # normalize_self_feat(mol_graphs, 'fr_imide')
    # # 10
    # normalize_self_feat(mol_graphs, 'Kappa2')
    # normalize_self_feat(mol_graphs, 'MinAbsPartialCharge')
    # normalize_self_feat(mol_graphs, 'NumAromaticHeterocycles')
    # # 20
    # normalize_self_feat(mol_graphs, 'SlogP_VSA1')
    # normalize_self_feat(mol_graphs, 'fr_amide')
    # normalize_self_feat(mol_graphs, 'BalabanJ')
    # normalize_self_feat(mol_graphs, 'fr_Ar_NH')
    # normalize_self_feat(mol_graphs, 'PEOE_VSA8')
    # normalize_self_feat(mol_graphs, 'NumSaturatedRings')
    # normalize_self_feat(mol_graphs, 'fr_NH0')
    # normalize_self_feat(mol_graphs, 'PEOE_VSA13')
    # normalize_self_feat(mol_graphs, 'fr_barbitur')
    # normalize_self_feat(mol_graphs, 'fr_alkyl_halide')
    # ####################################################

    return samples


atomic_props = read_atom_prop()
