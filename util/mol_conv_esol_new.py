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
dim_self_feat = 63


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
            mol_graph.MolLogP = dsc.MolLogP(mol)
            mol_graph.MaxAbsPartialCharge = dsc.MaxAbsPartialCharge(mol)
            mol_graph.MaxEStateIndex = dsc.MaxEStateIndex(mol)
            mol_graph.SMR_VSA10 = dsc.SMR_VSA10(mol)
            mol_graph.Kappa2 = dsc.Kappa2(mol)
            # 6
            mol_graph.BCUT2D_MWLOW = dsc.BCUT2D_MWLOW(mol)
            mol_graph.PEOE_VSA13 = dsc.PEOE_VSA13(mol)
            mol_graph.MinAbsPartialCharge = dsc.MinAbsPartialCharge(mol)
            mol_graph.BCUT2D_CHGHI = dsc.BCUT2D_CHGHI(mol)
            mol_graph.PEOE_VSA6 = dsc.PEOE_VSA6(mol)
            # 11
            mol_graph.SlogP_VSA1 = dsc.SlogP_VSA1(mol)
            mol_graph.fr_nitro = dsc.fr_nitro(mol)
            mol_graph.BalabanJ = dsc.BalabanJ(mol)
            mol_graph.SMR_VSA9 = dsc.SMR_VSA9(mol)
            mol_graph.fr_alkyl_halide = dsc.fr_alkyl_halide(mol)
            # 16
            mol_graph.fr_hdrzine = dsc.fr_hdrzine(mol)
            mol_graph.PEOE_VSA8 = dsc.PEOE_VSA8(mol)
            mol_graph.fr_Ar_NH = dsc.fr_Ar_NH(mol)
            mol_graph.fr_imidazole = dsc.fr_imidazole(mol)
            mol_graph.fr_Nhpyrrole = dsc.fr_Nhpyrrole(mol)
            # 21
            mol_graph.EState_VSA5 = dsc.EState_VSA5(mol)
            mol_graph.PEOE_VSA4 = dsc.PEOE_VSA4(mol)
            mol_graph.fr_ester = dsc.fr_ester(mol)
            mol_graph.PEOE_VSA2 = dsc.PEOE_VSA2(mol)
            mol_graph.NumAromaticCarbocycles = dsc.NumAromaticCarbocycles(mol)
            # 26
            mol_graph.BCUT2D_LOGPHI = dsc.BCUT2D_LOGPHI(mol)
            mol_graph.EState_VSA11 = dsc.EState_VSA11(mol)
            mol_graph.fr_furan = dsc.fr_furan(mol)
            mol_graph.EState_VSA2 = dsc.EState_VSA2(mol)
            mol_graph.fr_benzene = dsc.fr_benzene(mol)
            # 31
            mol_graph.fr_sulfide = dsc.fr_sulfide(mol)
            mol_graph.fr_aryl_methyl = dsc.fr_aryl_methyl(mol)
            mol_graph.SlogP_VSA10 = dsc.SlogP_VSA10(mol)
            mol_graph.HeavyAtomMolWt = dsc.HeavyAtomMolWt(mol)
            mol_graph.fr_nitro_arom_nonortho = dsc.fr_nitro_arom_nonortho(mol)
            # 36
            mol_graph.FpDensityMorgan2 = dsc.FpDensityMorgan2(mol)
            mol_graph.EState_VSA8 = dsc.EState_VSA8(mol)
            mol_graph.fr_bicyclic = dsc.fr_bicyclic(mol)
            mol_graph.fr_aniline = dsc.fr_aniline(mol)
            mol_graph.fr_allylic_oxid = dsc.fr_allylic_oxid(mol)
            # 41
            mol_graph.fr_C_S = dsc.fr_C_S(mol)
            mol_graph.SlogP_VSA7 = dsc.SlogP_VSA7(mol)
            mol_graph.SlogP_VSA4 = dsc.SlogP_VSA4(mol)
            mol_graph.fr_para_hydroxylation = dsc.fr_para_hydroxylation(mol)
            mol_graph.PEOE_VSA7 = dsc.PEOE_VSA7(mol)
            # 46
            mol_graph.fr_Al_OH_noTert = dsc.fr_Al_OH_noTert(mol)
            mol_graph.fr_pyridine = dsc.fr_pyridine(mol)
            mol_graph.fr_phos_acid = dsc.fr_phos_acid(mol)
            mol_graph.fr_phos_ester = dsc.fr_phos_ester(mol)
            mol_graph.NumAromaticHeterocycles = dsc.NumAromaticHeterocycles(mol)
            # 51
            mol_graph.EState_VSA7 = dsc.EState_VSA7(mol)
            mol_graph.PEOE_VSA12 = dsc.PEOE_VSA12(mol)
            mol_graph.Ipc = dsc.Ipc(mol)
            mol_graph.FpDensityMorgan1 = dsc.FpDensityMorgan1(mol)
            mol_graph.PEOE_VSA14 = dsc.PEOE_VSA14(mol)
            # 56
            mol_graph.fr_guanido = dsc.fr_guanido(mol)
            mol_graph.fr_benzodiazepine = dsc.fr_benzodiazepine(mol)
            mol_graph.fr_thiophene = dsc.fr_thiophene(mol)
            mol_graph.fr_Ndealkylation1 = dsc.fr_Ndealkylation1(mol)
            mol_graph.fr_aldehyde = dsc.fr_aldehyde(mol)
            # 61
            mol_graph.fr_term_acetylene = dsc.fr_term_acetylene(mol)
            mol_graph.SMR_VSA2 = dsc.SMR_VSA2(mol)
            mol_graph.fr_lactone = dsc.fr_lactone(mol)

            ####################################################

            samples.append((mol_graph, target[i]))
            mol_graphs.append(mol_graph)

    ####################################################
    # 1
    normalize_self_feat(mol_graphs, 'MolLogP')
    normalize_self_feat(mol_graphs, 'MaxAbsPartialCharge')
    normalize_self_feat(mol_graphs, 'MaxEStateIndex')
    normalize_self_feat(mol_graphs, 'SMR_VSA10')
    normalize_self_feat(mol_graphs, 'Kappa2')
    # 6
    normalize_self_feat(mol_graphs, 'BCUT2D_MWLOW')
    normalize_self_feat(mol_graphs, 'PEOE_VSA13')
    normalize_self_feat(mol_graphs, 'MinAbsPartialCharge')
    normalize_self_feat(mol_graphs, 'BCUT2D_CHGHI')
    normalize_self_feat(mol_graphs, 'PEOE_VSA6')
    # 11
    normalize_self_feat(mol_graphs, 'SlogP_VSA1')
    normalize_self_feat(mol_graphs, 'fr_nitro')
    normalize_self_feat(mol_graphs, 'BalabanJ')
    normalize_self_feat(mol_graphs, 'SMR_VSA9')
    normalize_self_feat(mol_graphs, 'fr_alkyl_halide')
    # 16
    normalize_self_feat(mol_graphs, 'fr_hdrzine')
    normalize_self_feat(mol_graphs, 'PEOE_VSA8')
    normalize_self_feat(mol_graphs, 'fr_Ar_NH')
    normalize_self_feat(mol_graphs, 'fr_imidazole')
    normalize_self_feat(mol_graphs, 'fr_Nhpyrrole')
    # 21
    normalize_self_feat(mol_graphs, 'EState_VSA5')
    normalize_self_feat(mol_graphs, 'PEOE_VSA4')
    normalize_self_feat(mol_graphs, 'fr_ester')
    normalize_self_feat(mol_graphs, 'PEOE_VSA2')
    normalize_self_feat(mol_graphs, 'NumAromaticCarbocycles')
    # 26
    normalize_self_feat(mol_graphs, 'BCUT2D_LOGPHI')
    normalize_self_feat(mol_graphs, 'EState_VSA11')
    normalize_self_feat(mol_graphs, 'fr_furan')
    normalize_self_feat(mol_graphs, 'EState_VSA2')
    normalize_self_feat(mol_graphs, 'fr_benzene')
    # 31
    normalize_self_feat(mol_graphs, 'fr_sulfide')
    normalize_self_feat(mol_graphs, 'fr_aryl_methyl')
    normalize_self_feat(mol_graphs, 'SlogP_VSA10')
    normalize_self_feat(mol_graphs, 'HeavyAtomMolWt')
    normalize_self_feat(mol_graphs, 'fr_nitro_arom_nonortho')
    # 36
    normalize_self_feat(mol_graphs, 'FpDensityMorgan2')
    normalize_self_feat(mol_graphs, 'EState_VSA8')
    normalize_self_feat(mol_graphs, 'fr_bicyclic')
    normalize_self_feat(mol_graphs, 'fr_aniline')
    normalize_self_feat(mol_graphs, 'fr_allylic_oxid')
    # 41
    normalize_self_feat(mol_graphs, 'fr_C_S')
    normalize_self_feat(mol_graphs, 'SlogP_VSA7')
    normalize_self_feat(mol_graphs, 'SlogP_VSA4')
    normalize_self_feat(mol_graphs, 'fr_para_hydroxylation')
    normalize_self_feat(mol_graphs, 'PEOE_VSA7')
    # 46
    normalize_self_feat(mol_graphs, 'fr_Al_OH_noTert')
    normalize_self_feat(mol_graphs, 'fr_pyridine')
    normalize_self_feat(mol_graphs, 'fr_phos_acid')
    normalize_self_feat(mol_graphs, 'fr_phos_ester')
    normalize_self_feat(mol_graphs, 'NumAromaticHeterocycles')
    # 51
    normalize_self_feat(mol_graphs, 'EState_VSA7')
    normalize_self_feat(mol_graphs, 'PEOE_VSA12')
    normalize_self_feat(mol_graphs, 'Ipc')
    normalize_self_feat(mol_graphs, 'FpDensityMorgan1')
    normalize_self_feat(mol_graphs, 'PEOE_VSA14')
    # 56
    normalize_self_feat(mol_graphs, 'fr_guanido')
    normalize_self_feat(mol_graphs, 'fr_benzodiazepine')
    normalize_self_feat(mol_graphs, 'fr_thiophene')
    normalize_self_feat(mol_graphs, 'fr_Ndealkylation1')
    normalize_self_feat(mol_graphs, 'fr_aldehyde')
    # 61
    normalize_self_feat(mol_graphs, 'fr_term_acetylene')
    normalize_self_feat(mol_graphs, 'SMR_VSA2')
    normalize_self_feat(mol_graphs, 'fr_lactone')
    return samples

atomic_props = read_atom_prop()