import pandas
import torch
import dgl
import numpy as np
import rdkit.Chem.Descriptors as dsc
from rdkit import Chem
from util import util
#from mendeleev import get_table
from mendeleev.fetch import fetch_table
import traceback

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
dim_self_feat = 59


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
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.MinEStateIndex = dsc.MinEStateIndex(mol)
            mol_graph.qed = dsc.qed(mol)
            mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            # 6
            mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            mol_graph.BCUT2D_CHGLO = dsc.BCUT2D_CHGLO(mol)
            mol_graph.BCUT2D_LOGPHI = dsc.BCUT2D_LOGPHI(mol)
            mol_graph.BCUT2D_MRHI = dsc.BCUT2D_MRHI(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            # 11
            mol_graph.BertzCT = dsc.BertzCT(mol)
            mol_graph.Chi0n = dsc.Chi0n(mol)
            mol_graph.Chi1 = dsc.Chi1(mol)
            mol_graph.Chi3n = dsc.Chi3n(mol)
            mol_graph.Chi4n = dsc.Chi4n(mol)
            # 16
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.PEOE_VSA1 = dsc.PEOE_VSA1(mol)
            mol_graph.PEOE_VSA14 = dsc.PEOE_VSA14(mol)
            mol_graph.PEOE_VSA2 = dsc.PEOE_VSA2(mol)
            mol_graph.PEOE_VSA3 = dsc.PEOE_VSA3(mol)
            # 21
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.PEOE_VSA5 = dsc.PEOE_VSA5(mol)
            mol_graph.PEOE_VSA6 = dsc.PEOE_VSA6(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            mol_graph.SMR_VSA1 = dsc.SMR_VSA1(mol)
            # 26
            mol_graph.SMR_VSA2 = dsc.SMR_VSA2(mol)
            mol_graph.SMR_VSA4 = dsc.SMR_VSA4(mol)
            mol_graph.SMR_VSA7 = dsc.SMR_VSA7(mol)
            mol_graph.SlogP_VSA11 = dsc.SlogP_VSA11(mol)
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            # 31
            mol_graph.SlogP_VSA6 = dsc.SlogP_VSA6(mol)
            mol_graph.TPSA = dsc.TPSA(mol)
            mol_graph.EState_VSA3 = dsc.EState_VSA3(mol)
            mol_graph.EState_VSA4 = dsc.EState_VSA4(mol)
            mol_graph.EState_VSA6 = dsc.EState_VSA6(mol)
            # 36
            mol_graph.EState_VSA8 = dsc.EState_VSA8(mol)
            mol_graph.EState_VSA9 = dsc.EState_VSA9(mol)
            mol_graph.VSA_EState4 = dsc.VSA_EState4(mol)
            mol_graph.VSA_EState6 = dsc.VSA_EState6(mol)
            mol_graph.NumAliphaticCarbocycles = dsc.NumAliphaticCarbocycles(mol)
            # 41
            mol_graph.NumAromaticRings = dsc.NumAromaticRings(mol)
            mol_graph.NumRotatableBonds = dsc.NumRotatableBonds(mol)
            mol_graph.NumSaturatedRings = dsc.NumSaturatedRings(mol)
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.MolMR = dsc.MolMR(mol)
            # 46
            mol_graph.fr_Al_COO = dsc.fr_Al_COO(mol)
            mol_graph.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.fr_NH0 = dsc.fr_NH0(mol)
            mol_graph.fr_NH1 = dsc.fr_NH1(mol)
            # 51
            mol_graph.fr_aldehyde = dsc.fr_aldehyde(mol)
            mol_graph.fr_amide = dsc.fr_amide(mol)
            mol_graph.fr_aniline = dsc.fr_aniline(mol)
            mol_graph.fr_ether = dsc.fr_ether(mol)
            mol_graph.fr_imide = dsc.fr_imide(mol)
            # 56
            mol_graph.fr_ketone = dsc.fr_ketone(mol)
            mol_graph.fr_ketone_Topliss = dsc.fr_ketone_Topliss(mol)
            mol_graph.fr_nitrile = dsc.fr_nitrile(mol)
            mol_graph.fr_oxime = dsc.fr_oxime(mol)
            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # 1
    normalize_self_feat(mol_graphs, 'MaxEStateIndex')
    normalize_self_feat(mol_graphs, 'MinEStateIndex')
    normalize_self_feat(mol_graphs, 'qed')
    normalize_self_feat(mol_graphs, 'MinAbsPartialCharge')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan2')
    # 6
    normalize_self_feat(mol_graphs, 'BCUT2D_MWLOW')
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGLO')
    normalize_self_feat(mol_graphs, 'BCUT2D_LOGPHI')
    normalize_self_feat(mol_graphs, 'BCUT2D_MRHI')
    normalize_self_feat(mol_graphs, 'BalabanJ')
    # 11
    normalize_self_feat(mol_graphs, 'BertzCT')
    normalize_self_feat(mol_graphs, 'Chi0n')
    normalize_self_feat(mol_graphs, 'Chi1')
    normalize_self_feat(mol_graphs, 'Chi3n')
    normalize_self_feat(mol_graphs, 'Chi4n')
    # 16
    normalize_self_feat(mol_graphs, 'Ipc')
    normalize_self_feat(mol_graphs, 'PEOE_VSA1')
    normalize_self_feat(mol_graphs, 'PEOE_VSA14')
    normalize_self_feat(mol_graphs, 'PEOE_VSA2')
    normalize_self_feat(mol_graphs, 'PEOE_VSA3')
    # 21
    normalize_self_feat(mol_graphs, 'PEOE_VSA4')
    normalize_self_feat(mol_graphs, 'PEOE_VSA5')
    normalize_self_feat(mol_graphs, 'PEOE_VSA6')
    normalize_self_feat(mol_graphs, 'PEOE_VSA7')
    normalize_self_feat(mol_graphs, 'SMR_VSA1')
    # 26
    normalize_self_feat(mol_graphs, 'SMR_VSA2')
    normalize_self_feat(mol_graphs, 'SMR_VSA4')
    normalize_self_feat(mol_graphs, 'SMR_VSA7')
    normalize_self_feat(mol_graphs, 'SlogP_VSA11')
    normalize_self_feat(mol_graphs, 'SlogP_VSA4')
    # 31
    normalize_self_feat(mol_graphs, 'SlogP_VSA6')
    normalize_self_feat(mol_graphs, 'TPSA')
    normalize_self_feat(mol_graphs, 'EState_VSA3')
    normalize_self_feat(mol_graphs, 'EState_VSA4')
    normalize_self_feat(mol_graphs, 'EState_VSA6')
    # 36
    normalize_self_feat(mol_graphs, 'EState_VSA8')
    normalize_self_feat(mol_graphs, 'EState_VSA9')
    normalize_self_feat(mol_graphs, 'VSA_EState4')
    normalize_self_feat(mol_graphs, 'VSA_EState6')
    normalize_self_feat(mol_graphs, 'NumAliphaticCarbocycles')
    # 41
    normalize_self_feat(mol_graphs, 'NumAromaticRings')
    normalize_self_feat(mol_graphs, 'NumRotatableBonds')
    normalize_self_feat(mol_graphs, 'NumSaturatedRings')
    normalize_self_feat(mol_graphs, 'MolLogP')
    normalize_self_feat(mol_graphs, 'MolMR')
    # 46
    normalize_self_feat(mol_graphs, 'fr_Al_COO')
    normalize_self_feat(mol_graphs, 'fr_Al_OH_noTert')
    normalize_self_feat(mol_graphs, 'fr_Ar_NH')
    normalize_self_feat(mol_graphs, 'fr_NH0')
    normalize_self_feat(mol_graphs, 'fr_NH1')
    # 51
    normalize_self_feat(mol_graphs, 'fr_aldehyde')
    normalize_self_feat(mol_graphs, 'fr_amide')
    normalize_self_feat(mol_graphs, 'fr_aniline')
    normalize_self_feat(mol_graphs, 'fr_ether')
    normalize_self_feat(mol_graphs, 'fr_imide')
    # 56
    normalize_self_feat(mol_graphs, 'fr_ketone')
    normalize_self_feat(mol_graphs, 'fr_ketone_Topliss')
    normalize_self_feat(mol_graphs, 'fr_nitrile')
    normalize_self_feat(mol_graphs, 'fr_oxime')
    ####################################################

    return samples

atomic_props = read_atom_prop()