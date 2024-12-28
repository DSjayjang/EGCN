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
dim_self_feat = 72


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
            mol_graph.Chi0n = dsc.Chi0n(mol)
            mol_graph.Chi0 = dsc.Chi0(mol)
            mol_graph.Kappa1 = dsc.Kappa1(mol)
            mol_graph.RingCount = dsc.RingCount(mol)
            mol_graph.Chi4n = dsc.Chi4n(mol)
            # 6
            mol_graph.NHOHCount = dsc.NHOHCount(mol)
            mol_graph.SMR_VSA7 = dsc.SMR_VSA7(mol)
            mol_graph.FractionCSP3 = dsc.FractionCSP3(mol)
            mol_graph.NumSaturatedCarbocycles = dsc.NumSaturatedCarbocycles(mol)
            mol_graph.Chi1 = dsc.Chi1(mol)
            # 11
            mol_graph.BertzCT = dsc.BertzCT(mol)
            mol_graph.fr_Al_OH = dsc.fr_Al_OH(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            mol_graph.BCUT2D_LOGPLOW = dsc.BCUT2D_LOGPLOW(mol)
            # 16
            mol_graph.Chi2n = dsc.Chi2n(mol)
            mol_graph.VSA_EState7 = dsc.VSA_EState7(mol)
            mol_graph.NumRotatableBonds = dsc.NumRotatableBonds(mol)
            mol_graph.BCUT2D_LOGPHI = dsc.BCUT2D_LOGPHI(mol)
            mol_graph.fr_NH1 = dsc.fr_NH1(mol)
            # 21
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            mol_graph.fr_ether = dsc.fr_ether(mol)
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            # 26
            mol_graph.fr_NH0 = dsc.fr_NH0(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.NumSaturatedHeterocycles = dsc.NumSaturatedHeterocycles(mol)
            mol_graph.BCUT2D_CHGLO = dsc.BCUT2D_CHGLO(mol)
            # 31
            mol_graph.NumAromaticRings = dsc.NumAromaticRings(mol)
            mol_graph.VSA_EState8 = dsc.VSA_EState8(mol)
            mol_graph.VSA_EState4 = dsc.VSA_EState4(mol)
            mol_graph.VSA_EState2 = dsc.VSA_EState2(mol)
            mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            # 36
            mol_graph.SMR_VSA6 = dsc.SMR_VSA6(mol)
            mol_graph.fr_C_O = dsc.fr_C_O(mol)
            mol_graph.VSA_EState6 = dsc.VSA_EState6(mol)
            mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            mol_graph.fr_Ar_OH = dsc.fr_Ar_OH(mol)
            # 41
            mol_graph.SMR_VSA5 = dsc.SMR_VSA5(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.fr_Al_COO = dsc.fr_Al_COO(mol)
            mol_graph.fr_NH2 = dsc.fr_NH2(mol)
            mol_graph.VSA_EState5 = dsc.VSA_EState5(mol)
            # 46
            mol_graph.fr_epoxide = dsc.fr_epoxide(mol)
            mol_graph.fr_term_acetylene = dsc.fr_term_acetylene(mol)
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.fr_oxime = dsc.fr_oxime(mol)
            mol_graph.fr_aldehyde = dsc.fr_aldehyde(mol)
            # 51
            mol_graph.EState_VSA9 = dsc.EState_VSA9(mol)
            mol_graph.SlogP_VSA3 = dsc.SlogP_VSA3(mol)
            mol_graph.fr_quatN = dsc.fr_quatN(mol)
            mol_graph.FpDensityMorgan3 = dsc.FpDensityMorgan3(mol)
            mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            # 56
            mol_graph.fr_Ar_N = dsc.fr_Ar_N(mol)
            mol_graph.EState_VSA1 = dsc.EState_VSA1(mol)
            mol_graph.fr_nitrile = dsc.fr_nitrile(mol)
            mol_graph.PEOE_VSA3 = dsc.PEOE_VSA3(mol)
            mol_graph.fr_piperdine = dsc.fr_piperdine(mol)
            # 61
            mol_graph.fr_aryl_methyl = dsc.fr_aryl_methyl(mol)
            mol_graph.fr_nitro = dsc.fr_nitro(mol)
            mol_graph.PEOE_VSA2 = dsc.PEOE_VSA2(mol)
            mol_graph.EState_VSA2 = dsc.EState_VSA2(mol)
            mol_graph.fr_unbrch_alkane = dsc.fr_unbrch_alkane(mol)
            # 66
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            mol_graph.qed = dsc.qed(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.PEOE_VSA5 = dsc.PEOE_VSA5(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            # 71
            mol_graph.fr_bicyclic = dsc.fr_bicyclic(mol)
            mol_graph.EState_VSA7 = dsc.EState_VSA7(mol)
            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # 1
    normalize_self_feat(mol_graphs, 'Chi0n')
    normalize_self_feat(mol_graphs, 'Chi0')
    normalize_self_feat(mol_graphs, 'Kappa1')
    normalize_self_feat(mol_graphs, 'RingCount')
    normalize_self_feat(mol_graphs, 'Chi4n')
    # 6
    normalize_self_feat(mol_graphs, 'NHOHCount')
    normalize_self_feat(mol_graphs, 'SMR_VSA7')
    normalize_self_feat(mol_graphs, 'FractionCSP3')
    normalize_self_feat(mol_graphs, 'NumSaturatedCarbocycles')
    normalize_self_feat(mol_graphs, 'Chi1')
    # 11
    normalize_self_feat(mol_graphs, 'BertzCT')
    normalize_self_feat(mol_graphs, 'fr_Al_OH')
    normalize_self_feat(mol_graphs, 'BalabanJ')
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGHI')
    normalize_self_feat(mol_graphs, 'BCUT2D_LOGPLOW')
    # 16
    normalize_self_feat(mol_graphs, 'Chi2n')
    normalize_self_feat(mol_graphs, 'VSA_EState7')
    normalize_self_feat(mol_graphs, 'NumRotatableBonds')
    normalize_self_feat(mol_graphs, 'BCUT2D_LOGPHI')
    normalize_self_feat(mol_graphs, 'fr_NH1')
    # 21
    normalize_self_feat(mol_graphs, 'FpDensityMorgan1')
    normalize_self_feat(mol_graphs, 'fr_ether')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan2')
    normalize_self_feat(mol_graphs, 'Kappa2')
    normalize_self_feat(mol_graphs, 'SMR_VSA9')
    # 26
    normalize_self_feat(mol_graphs, 'fr_NH0')
    normalize_self_feat(mol_graphs, 'SMR_VSA10')
    normalize_self_feat(mol_graphs, 'fr_Ar_NH')
    normalize_self_feat(mol_graphs, 'NumSaturatedHeterocycles')
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGLO')
    # 31
    normalize_self_feat(mol_graphs, 'NumAromaticRings')
    normalize_self_feat(mol_graphs, 'VSA_EState8')
    normalize_self_feat(mol_graphs, 'VSA_EState4')
    normalize_self_feat(mol_graphs, 'VSA_EState2')
    normalize_self_feat(mol_graphs, 'SlogP_VSA1')
    # 36
    normalize_self_feat(mol_graphs, 'SMR_VSA6')
    normalize_self_feat(mol_graphs, 'fr_C_O')
    normalize_self_feat(mol_graphs, 'VSA_EState6')
    normalize_self_feat(mol_graphs, 'BCUT2D_MWLOW')
    normalize_self_feat(mol_graphs, 'fr_Ar_OH')
    # 41
    normalize_self_feat(mol_graphs, 'SMR_VSA5')
    normalize_self_feat(mol_graphs, 'PEOE_VSA4')
    normalize_self_feat(mol_graphs, 'fr_Al_COO')
    normalize_self_feat(mol_graphs, 'fr_NH2')
    normalize_self_feat(mol_graphs, 'VSA_EState5')
    # 46
    normalize_self_feat(mol_graphs, 'fr_epoxide')
    normalize_self_feat(mol_graphs, 'fr_term_acetylene')
    normalize_self_feat(mol_graphs, 'Ipc')
    normalize_self_feat(mol_graphs, 'fr_oxime')
    normalize_self_feat(mol_graphs, 'fr_aldehyde')
    # 51
    normalize_self_feat(mol_graphs, 'EState_VSA9')
    normalize_self_feat(mol_graphs, 'SlogP_VSA3')
    normalize_self_feat(mol_graphs, 'fr_quatN')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan3')
    normalize_self_feat(mol_graphs, 'PEOE_VSA8')
    # 56
    normalize_self_feat(mol_graphs, 'fr_Ar_N')
    normalize_self_feat(mol_graphs, 'EState_VSA1')
    normalize_self_feat(mol_graphs, 'fr_nitrile')
    normalize_self_feat(mol_graphs, 'PEOE_VSA3')
    normalize_self_feat(mol_graphs, 'fr_piperdine')
    # 61
    normalize_self_feat(mol_graphs, 'fr_aryl_methyl')
    normalize_self_feat(mol_graphs, 'fr_nitro')
    normalize_self_feat(mol_graphs, 'PEOE_VSA2')
    normalize_self_feat(mol_graphs, 'EState_VSA2')
    normalize_self_feat(mol_graphs, 'fr_unbrch_alkane')
    # 66
    normalize_self_feat(mol_graphs, 'fr_pyridine')
    normalize_self_feat(mol_graphs, 'qed')
    normalize_self_feat(mol_graphs, 'SlogP_VSA10')
    normalize_self_feat(mol_graphs, 'PEOE_VSA5')
    normalize_self_feat(mol_graphs, 'PEOE_VSA7')
    # 71
    normalize_self_feat(mol_graphs, 'fr_bicyclic')
    normalize_self_feat(mol_graphs, 'EState_VSA7')
    ####################################################

    return samples


atomic_props = read_atom_prop()