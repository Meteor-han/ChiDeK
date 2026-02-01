"""
Shared data processing utilities for central and axial chirality tasks.
"""
import os
import pickle
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm import tqdm

from utils import getNodeFeatures


# ============== Central Chirality: Shared Helpers ==============

def get_coords(mol):
    """Extract 3D coordinates from molecule conformer."""
    n_atoms = mol.GetNumAtoms()
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=np.float32)


def build_atom_types_and_edges(mol, chiral_idx):
    """
    Build atom_types (0=chiral_center, 1=chiral_related, 2=non_related) and edge_types.
    Returns (atom_types, edge_types).
    """
    n_atoms = mol.GetNumAtoms()
    atom_types = np.ones(n_atoms, dtype=np.int64) * 2
    atom_types[chiral_idx] = 0

    adj = rdmolops.GetAdjacencyMatrix(mol)
    for idx_c in chiral_idx:
        neighbors = np.where(adj[idx_c] > 0)[0]
        for nb in neighbors:
            if atom_types[nb] == 2:
                atom_types[nb] = 1

    edge_types = np.full((n_atoms, n_atoms), 0, dtype=np.int64)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            if (atom_types[i] == 0 and atom_types[j] == 1) or (atom_types[i] == 1 and atom_types[j] == 0):
                edge_types[i, j] = 1
            elif (atom_types[i] == 0 and atom_types[j] == 2) or (atom_types[i] == 2 and atom_types[j] == 0):
                edge_types[i, j] = 2

    return atom_types, edge_types


def build_central_features(mol, chiral_idx, res_entry, use_node_features=True):
    """
    Build core feature dict for central chirality: coords, atom_types, edge_types, atom_onehot, atom_chiral.
    res_entry: ChiralFinder result for this molecule (with "center id" and "quadrupole matrix").
    """
    n_atoms = mol.GetNumAtoms()
    coords = get_coords(mol)
    atom_types, edge_types = build_atom_types_and_edges(mol, chiral_idx)

    if use_node_features:
        atoms = Chem.rdchem.Mol.GetAtoms(mol)
        atom_onehot = getNodeFeatures(atoms, mol, False)
    else:
        atom_onehot = np.eye(100, dtype=np.float32)[[atom.GetAtomicNum() % 100 for atom in mol.GetAtoms()]]

    atom_chiral = np.zeros((n_atoms, 9), dtype=np.float32)
    for i in range(len(res_entry["center id"])):
        cid = res_entry["center id"][i]
        if cid in chiral_idx:
            atom_chiral[cid] = np.array(res_entry["quadrupole matrix"][i][0].reshape(9), dtype=np.float32)

    return {
        "coords": coords,
        "atom_types": atom_types,
        "edge_types": edge_types,
        "atom_onehot": atom_onehot,
        "atom_chiral": atom_chiral,
    }


def compute_chiral_matrix_from_coords(mol, chiral_idx, center_i):
    """Compute 9-dim chiral matrix from coordinates (fallback when center not in ChiralFinder res)."""
    neigh_cor = []
    atom = mol.GetAtomWithIdx(chiral_idx[center_i])
    neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
    for nb in neighbors:
        pos = mol.GetConformer().GetAtomPosition(nb)
        neigh_cor.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))
    pos = mol.GetConformer().GetAtomPosition(chiral_idx[center_i])
    neigh_cor.insert(0, np.array([pos.x, pos.y, pos.z], dtype=np.float32))
    if len(neigh_cor) == 4:
        neigh_cor.insert(1, (neigh_cor[1] + neigh_cor[2] + neigh_cor[3] - neigh_cor[0] * 3) / 3 * -1.0 + neigh_cor[0])
    elif len(neigh_cor) < 4:
        for _ in range(5 - len(neigh_cor)):
            neigh_cor.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    a = neigh_cor[1] - neigh_cor[0]
    b = neigh_cor[4] - neigh_cor[3]
    c = neigh_cor[4] - neigh_cor[2]
    mat = np.array([a, b, c])
    return mat.reshape(9).astype(np.float32)


def run_chiral_finder_central(mols, batch_size=500, n_cpus=4):
    """Run ChiralFinder.get_central on molecules in batches."""
    from chiralfinder import ChiralFinder
    res = []
    for start in range(0, len(mols), batch_size):
        end = start + batch_size
        batch_mols = mols[start:end]
        cf_batch = ChiralFinder(batch_mols, "molecules")
        res.extend(cf_batch.get_central(n_cpus=n_cpus))
    return res


# ============== Central Chirality: Task-specific Processors ==============

def process_central_ecd(data, disable_tqdm=False):
    """Process data for central ECD spectrum prediction. data: DataFrame or list of dict-like rows."""
    data_list = []
    rows = list(data.iterrows() if hasattr(data, 'iterrows') else enumerate(data))
    mols = [Chem.AddHs(row['rdkit_mol'], addCoords=True) for _, row in tqdm(rows, desc="processing chiral", disable=disable_tqdm)]
    res = run_chiral_finder_central(mols)

    for idx, (_, row) in enumerate(tqdm(rows, desc="processing data", disable=disable_tqdm)):
        mol = mols[idx]  # Use same mol (with Hs) as ChiralFinder
        chiral_idx = res[idx]["center id"]
        if not chiral_idx:
            print(f"Warning: no chiral center in molecule {idx}, SMILES: {Chem.MolToSmiles(mol)}")
            continue

        feat = build_central_features(mol, chiral_idx, res[idx], use_node_features=True)
        feat["label_num"] = int(row['peak_num'])
        feat["label_position"] = np.array(row['peak_position'], dtype=np.int8)
        feat["label_height"] = np.array(row['peak_height'], dtype=np.int8)
        feat["rdkit_mol"] = mol
        data_list.append(feat)
    return data_list


def process_central_ranking(df, disable_tqdm=False):
    """Process DataFrame for enantiomer ranking. Returns dict keyed by df.index."""
    mols = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="processing chiral", disable=disable_tqdm):
        mol = row['rdkit_mol_cistrans_stereo']
        mol = Chem.AddHs(mol, addCoords=True)
        mols.append(mol)

    res = run_chiral_finder_central(mols)
    df_index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
    data_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="processing data", disable=disable_tqdm):
        pos = df_index_to_pos[idx]
        mol = row['rdkit_mol_cistrans_stereo']
        chiral_idx = [i for i, _ in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)]
        feat = build_central_features(mol, chiral_idx, res[pos], use_node_features=False)
        feat["label"] = float(row['top_score'])
        data_list.append((idx, feat))

    return dict(data_list)


def process_central_RS(df, tag, random_ratio, data_dir, disable_tqdm=False):
    """Process DataFrame for R/S classification with optional chiral center perturbation."""
    mols = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="processing chiral", disable=disable_tqdm):
        mol = row['rdkit_mol_cistrans_stereo']
        mol = Chem.AddHs(mol, addCoords=True)
        mols.append(mol)

    pth = os.path.join(data_dir, f"RS_chiralfinder_{tag}.pkl")
    if os.path.exists(pth):
        with open(pth, "rb") as f:
            res = pickle.load(f)
    else:
        res = run_chiral_finder_central(mols)
        os.makedirs(data_dir, exist_ok=True)
        with open(pth, "wb") as f:
            pickle.dump(res, f)

    n_random = int(len(df) * random_ratio)
    random_idx = set(random.sample(range(len(df)), n_random)) if n_random > 0 else set()
    df_pos_to_idx = list(df.index)
    data_list = []

    for pos, (idx, row) in enumerate(tqdm([x for x in df.iterrows()], total=len(df), desc="processing data", disable=disable_tqdm)):
        mol = row['rdkit_mol_cistrans_stereo']
        chiral_idx = sorted([i for i, _ in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)])
        n_atoms = mol.GetNumAtoms()
        non_chiral = [i for i in range(n_atoms) if i not in chiral_idx]

        if pos in random_idx and len(chiral_idx) > 0:
            r = random.random()
            if r < 0.5:
                drop_idx = random.choice(chiral_idx)
                chiral_idx = [c for c in chiral_idx if c != drop_idx]
            else:
                drop_idx = random.choice(chiral_idx)
                chiral_idx = [c for c in chiral_idx if c != drop_idx]
                if non_chiral:
                    chiral_idx.append(random.choice(non_chiral))

        feat = build_central_features(mol, chiral_idx, res[pos], use_node_features=True)
        for i, cid in enumerate(chiral_idx):
            if cid not in res[pos]["center id"]:
                feat["atom_chiral"][cid] = compute_chiral_matrix_from_coords(mol, chiral_idx, i)
        feat["label"] = float(row['RS_label_binary'])
        feat["rdkit_mol"] = mol
        data_list.append(feat)
    return data_list


# ============== Axial Chirality: Shared Helpers ==============

def _normalize_pairs(pairs):
    return {tuple(sorted(p)) for p in pairs}


def _compute_coverage_iou(label, pred):
    label_set = _normalize_pairs(label)
    pred_set = _normalize_pairs(pred)
    inter = label_set & pred_set
    union = label_set | pred_set
    coverage = 1 if inter == label_set else 0
    iou = len(inter) / len(union) if union else 1.0
    return coverage, iou


def _compute_axial_chiral_from_neighbors(mol, one_axial, coords, atom_types, edge_types, atom_chiral, n_atoms):
    """Fallback: compute chiral matrix from neighbor coordinates when not in ChiralFinder res."""
    neigh_set = set()
    for one_chiral_id in one_axial:
        neighbors = mol.GetAtomWithIdx(one_chiral_id).GetNeighbors()
        for x in neighbors:
            if x.GetIdx() not in one_axial:
                neigh_set.add(x.GetIdx())
    neigh_list = sorted(neigh_set)[:4]
    if len(neigh_list) < 3:
        return coords
    center = np.mean([coords[x] for x in one_axial], axis=0)
    neigh_cor = [center] + [coords[x] for x in neigh_list]
    if len(neigh_list) == 3:
        neigh_cor.append(-np.mean([coords[x] for x in neigh_list], axis=0))
    a = neigh_cor[1] - neigh_cor[0]
    b = neigh_cor[2] - neigh_cor[0]
    c = neigh_cor[3] - neigh_cor[4]
    mat = np.array([a, b, c])
    for j in one_axial:
        atom_chiral[j] = mat.reshape(9)
    for nb in neigh_list:
        if nb < n_atoms and atom_types[nb] == 2:
            atom_types[nb] = 1
            for oc in one_axial:
                edge_types[nb, oc] = 1
                edge_types[oc, nb] = 1
    return coords


# ============== Axial Chirality: Task-specific Processors ==============

def process_axial_chirality(data, random_ratio, data_dir, label_mode="ecd", disable_tqdm=False):
    """
    Process axial chirality data. label_mode: "ecd" (peak num/pos/height) or "optical" (binary label).
    """
    res_path = os.path.join(data_dir, "hct_ecd_axial_res.pkl")
    labels_path = os.path.join(data_dir, "axial_650.xlsx")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"ChiralFinder results not found at {res_path}.")
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    labels = pd.read_excel(labels_path)

    id2optical = None
    if label_mode == "optical":
        optical_path = os.path.join(data_dir, "optical_rotation_589nm.csv")
        optical_labels = pd.read_csv(optical_path)
        id2optical = {int(optical_labels["id"][i]): float(optical_labels["OR_589nm"][i]) for i in range(len(optical_labels))}

    coverage_scores, iou_scores = [], []
    data_list = []
    rows = [(i, r) for i, (_, r) in enumerate(data.iterrows())] if hasattr(data, 'iterrows') else list(enumerate(data))

    for idx, row in tqdm(rows, desc="processing data", disable=disable_tqdm):
        one_label_id = abs(int(row["id"].split("_")[-1]))
        one_label = eval(labels["label"][one_label_id])
        one_pred = res[idx]["chiral axes"]
        coverage, iou = _compute_coverage_iou(one_label, one_pred)
        coverage_scores.append(coverage)
        iou_scores.append(iou)
        chiral_type = labels["chiral_type"][one_label_id]

        mol = row['rdkit_mol']
        n_atoms = mol.GetNumAtoms()
        if random_ratio > 0.05:
            one_label = res[idx]["chiral axes"]
        chiral_idx = [x for t in one_label for x in t]
        if not chiral_idx:
            print(f"Warning: no chiral axis in molecule {idx}, SMILES: {Chem.MolToSmiles(mol)}")

        coords = get_coords(mol)
        atom_types = np.ones(n_atoms, dtype=np.int64) * 2
        atom_types[chiral_idx] = 0
        edge_types = np.full((n_atoms, n_atoms), 0, dtype=np.int64)
        atoms = Chem.rdchem.Mol.GetAtoms(mol)
        atom_onehot = getNodeFeatures(atoms, mol, False)
        atom_chiral = np.zeros((n_atoms, 9), dtype=np.float32)

        tag = False
        for one_axial in one_label:
            for i in range(len(res[idx]["chiral axes"])):
                if set(one_axial) == set(res[idx]["chiral axes"][i]):
                    tag = True
                    neighbors = res[idx]["neighbor ids"][i]
                    if isinstance(neighbors[0], list):
                        neighbors = sum(neighbors, [])
                    for nb in neighbors:
                        if nb < n_atoms and atom_types[nb] == 2:
                            atom_types[nb] = 1
                            for oc in one_axial:
                                edge_types[nb, oc] = 1
                                edge_types[oc, nb] = 1
                    qm = res[idx]["quadrupole matrix"][i]
                    axis_atoms = res[idx]["chiral axes"][i]
                    if len(qm) == len(axis_atoms):
                        for k, j in enumerate(axis_atoms):
                            atom_chiral[j] = np.array(np.array(qm[k]).reshape(9), dtype=np.float32)
                    else:
                        for j in axis_atoms:
                            atom_chiral[j] = np.array(np.array(qm[0]).reshape(9), dtype=np.float32)
                    break

        if not tag:
            mol = rdmolops.AddHs(mol, addCoords=True)
            coords = np.array([list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
            for one_axial in one_label:
                _compute_axial_chiral_from_neighbors(
                    mol, one_axial, coords, atom_types, edge_types, atom_chiral, n_atoms
                )

        item = {
            'coords': coords,
            'atom_types': atom_types,
            'edge_types': edge_types,
            'atom_onehot': atom_onehot,
            'atom_chiral': atom_chiral,
            'rdkit_mol': mol,
            'chiral_type': chiral_type,
        }
        if label_mode == "ecd":
            item["label_num"] = int(row['peak_num'])
            item["label_position"] = np.array(row['peak_position'], dtype=np.int8)
            item["label_height"] = np.array(row['peak_height'], dtype=np.int8)
        else:
            id_wo_abs_str = row["id"].split("_")[-1]
            if "-" in id_wo_abs_str:
                item["label"] = 1 - (1 if id2optical[one_label_id] > 0 else 0)
            else:
                item["label"] = 1 if id2optical[one_label_id] > 0 else 0
        data_list.append(item)

    print(f"Average Coverage: {np.mean(coverage_scores):.4f}, average IoU: {np.mean(iou_scores):.4f}")
    return data_list
