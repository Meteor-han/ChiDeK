import os
import sys
import torch
import argparse
import numpy as np
from typing import List
import rdkit
import rdkit.Chem
from sklearn.metrics import roc_auc_score
from rdkit import RDLogger

# Project root (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RDLogger.DisableLog("rdApp.*")

def compute_accuracy(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def compute_accuracy_multiclasses(preds, labels):
    """
    Compute accuracy for multi-class classification.

    Args:
        preds: Tensor of shape [N, C], model logits or probabilities.
        labels: Tensor of shape [N], ground-truth class indices.

    Returns:
        accuracy: float, average accuracy.
    """
    pred_classes = torch.argmax(preds, dim=1)
    correct = (pred_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total

def compute_accuracy_multitasks(logits, labels):
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).sum().item()
    return correct / labels.numel()

def compute_multitask_metrics(logits, labels):
    """
    logits: [B, T, 2] or [B, T]
    labels: [B, T] (0/1)
    """
    # if logits [B, T, 2]
    if logits.dim() == 3:
        logits = logits[..., 1]  # [B, T]

    labels = labels.float()
    probs = logits.numpy()
    labels_np = labels.numpy()
    if labels_np.ndim == 1:
        labels_np = labels_np[:, np.newaxis]
        probs = probs[:, np.newaxis]

    # ROC-AUC per task
    auc_list = []
    for i in range(labels_np.shape[1]):
        y_true = labels_np[:, i]
        y_pred = probs[:, i]
        if np.unique(y_true).size < 2:
            print(f"Task {i} has only one class present in y_true. Skipping AUC computation for this task.")
            continue
        auc = roc_auc_score(y_true, y_pred)
        auc_list.append(auc)

    mean_auc = np.mean(auc_list) if len(auc_list) > 0 else np.nan
    return mean_auc, auc_list

def collate_hct(batch: List[dict]):
    """
    Constructs padded tensors for:
      - queries: chiral centers across molecules -> pad to max_q
      - keys/vals: other atoms across molecules -> pad to max_k
    Returns a dict with:
      coords_q: (B, max_q, 3)
      coords_k: (B, max_k, 3)
      feats_q:  (B, max_q, F)
      feats_k:  (B, max_k, F)
      q_mask:   (B, max_q) True for real, False for pad
      k_mask:   (B, max_k) True for real, False for pad
      k_atom_types: (B, max_k) ints {1,2} telling whether chiral_related or non_related (0 should not appear here)
      q_counts, k_counts
    """
    batch_size = len(batch)
    coords_q_list, coords_k_list = [], []
    feats_q_list, feats_k_list = [], []
    feats_q_list_kv = []
    k_types_list = []
    q_counts, k_counts = [], []
    labels = []
    edge_types_qk_list = []
    weights = []

    for item in batch:
        coords = item['coords']
        types = item['atom_types']
        edge_types = item['edge_types']
        onehot = item['atom_onehot']
        chiral = item['atom_chiral']
        label = item['label']
        if 'weight' in item:
            weights.append(item['weight'])

        q_idx = np.where(types == 0)[0]
        k_idx = np.where(types != 0)[0]

        q_counts.append(len(q_idx))
        k_counts.append(len(k_idx))

        coords_q_list.append(coords[q_idx] if len(q_idx)>0 else np.zeros((0,3),dtype=np.float32))
        coords_k_list.append(coords[k_idx] if len(k_idx)>0 else np.zeros((0,3),dtype=np.float32))
        feats_q_list.append(chiral[q_idx] if len(q_idx)>0 else np.zeros((0,9),dtype=np.float32))
        feats_q_list_kv.append(onehot[q_idx] if len(q_idx)>0 else np.zeros((0,onehot.shape[1]),dtype=np.float32))
        feats_k_list.append(onehot[k_idx] if len(k_idx)>0 else np.zeros((0,onehot.shape[1]),dtype=np.float32))
        k_types_list.append(types[k_idx] if len(k_idx)>0 else np.zeros((0,),dtype=np.int64))
        # edge_types for each q and k
        edge_types_qk = edge_types[np.ix_(q_idx, k_idx)] if (len(q_idx)>0 and len(k_idx)>0) else np.full((len(q_idx), len(k_idx)), -1, dtype=np.int64)
        edge_types_qk_list.append(edge_types_qk)
        labels.append(label)

    max_q = max(max(q_counts),1)
    max_k = max(k_counts)
    F_q = feats_q_list[0].shape[1]
    F_k = feats_k_list[0].shape[1]

    coords_q = np.zeros((batch_size, max_q, 3), dtype=np.float32)
    feats_q = np.zeros((batch_size, max_q, F_q), dtype=np.float32)
    feats_q_kv = np.zeros((batch_size, max_q, F_k), dtype=np.float32)
    q_mask = np.zeros((batch_size, max_q), dtype=np.bool_)

    coords_k = np.zeros((batch_size, max_k, 3), dtype=np.float32)
    feats_k = np.zeros((batch_size, max_k, F_k), dtype=np.float32)
    k_mask = np.zeros((batch_size, max_k), dtype=np.bool_)
    k_atom_types = np.zeros((batch_size, max_k), dtype=np.int64)
    edge_types_qk = np.full((batch_size, max_q, max_k), 0, dtype=np.int64)

    for i in range(batch_size):
        nq = q_counts[i]
        nk = k_counts[i]
        if nq>0:
            coords_q[i,:nq,:] = coords_q_list[i]
            feats_q[i,:nq,:] = feats_q_list[i]
            feats_q_kv[i,:nq,:] = feats_q_list_kv[i]
            q_mask[i,:nq] = True
        if nk>0:
            coords_k[i,:nk,:] = coords_k_list[i]
            feats_k[i,:nk,:] = feats_k_list[i]
            k_mask[i,:nk] = True
            k_atom_types[i,:nk] = k_types_list[i]
        # pad edge_types_qk
        if nq>0 and nk>0:
            edge_types_qk[i,:nq,:nk] = edge_types_qk_list[i]

    return {
        'coords_q': torch.tensor(coords_q),
        'coords_k': torch.tensor(coords_k),
        'feats_q': torch.tensor(feats_q),
        'feats_q_kv': torch.tensor(feats_q_kv),
        'feats_k': torch.tensor(feats_k),
        'q_mask': torch.tensor(q_mask),
        'k_mask': torch.tensor(k_mask),
        'k_atom_types': torch.tensor(k_atom_types),
        'edge_types_qk': torch.tensor(edge_types_qk),
        'labels': torch.tensor(labels, dtype=torch.float32),
        'q_counts': torch.tensor(q_counts),
        'k_counts': torch.tensor(k_counts),
        'weights': torch.tensor(weights, dtype=torch.float32) if len(weights)>0 else None
    }


def collate_hct_ecd(batch: List[dict]):
    """
    Constructs padded tensors for:
      - queries: chiral centers across molecules -> pad to max_q
      - keys/vals: other atoms across molecules -> pad to max_k
    Returns a dict with:
      coords_q: (B, max_q, 3)
      coords_k: (B, max_k, 3)
      feats_q:  (B, max_q, F)
      feats_k:  (B, max_k, F)
      q_mask:   (B, max_q) True for real, False for pad
      k_mask:   (B, max_k) True for real, False for pad
      k_atom_types: (B, max_k) ints {1,2} telling whether chiral_related or non_related (0 should not appear here)
      q_counts, k_counts
    """
    batch_size = len(batch)
    coords_q_list, coords_k_list = [], []
    feats_q_list, feats_k_list = [], []
    feats_q_list_kv = []
    k_types_list = []
    q_counts, k_counts = [], []
    labels_nums = []
    labels_position = []
    labels_height = []
    edge_types_qk_list = []

    for item in batch:
        coords = item['coords']
        types = item['atom_types']
        edge_types = item['edge_types']
        onehot = item['atom_onehot']
        chiral = item['atom_chiral']
        label_num = item['label_num']
        label_position = item['label_position']
        label_height = item['label_height']

        q_idx = np.where(types == 0)[0]
        k_idx = np.where(types != 0)[0]

        q_counts.append(len(q_idx))
        k_counts.append(len(k_idx))

        coords_q_list.append(coords[q_idx] if len(q_idx)>0 else np.zeros((0,3),dtype=np.float32))
        coords_k_list.append(coords[k_idx] if len(k_idx)>0 else np.zeros((0,3),dtype=np.float32))
        feats_q_list.append(chiral[q_idx] if len(q_idx)>0 else np.zeros((0,9),dtype=np.float32))
        feats_q_list_kv.append(onehot[q_idx] if len(q_idx)>0 else np.zeros((0,onehot.shape[1]),dtype=np.float32))
        feats_k_list.append(onehot[k_idx] if len(k_idx)>0 else np.zeros((0,onehot.shape[1]),dtype=np.float32))
        k_types_list.append(types[k_idx] if len(k_idx)>0 else np.zeros((0,),dtype=np.int64))
        # edge_types for each q and k
        edge_types_qk = edge_types[np.ix_(q_idx, k_idx)] if (len(q_idx)>0 and len(k_idx)>0) else np.full((len(q_idx), len(k_idx)), -1, dtype=np.int64)
        edge_types_qk_list.append(edge_types_qk)
        labels_nums.append(label_num)
        labels_position.append(label_position)
        labels_height.append(label_height)

    max_q = max(max(q_counts),1)
    max_k = max(k_counts)
    F_q = feats_q_list[0].shape[1]
    F_k = feats_k_list[0].shape[1]

    coords_q = np.zeros((batch_size, max_q, 3), dtype=np.float32)
    feats_q = np.zeros((batch_size, max_q, F_q), dtype=np.float32)
    feats_q_kv = np.zeros((batch_size, max_q, F_k), dtype=np.float32)
    q_mask = np.zeros((batch_size, max_q), dtype=np.bool_)

    coords_k = np.zeros((batch_size, max_k, 3), dtype=np.float32)
    feats_k = np.zeros((batch_size, max_k, F_k), dtype=np.float32)
    k_mask = np.zeros((batch_size, max_k), dtype=np.bool_)
    k_atom_types = np.zeros((batch_size, max_k), dtype=np.int64)
    edge_types_qk = np.full((batch_size, max_q, max_k), 0, dtype=np.int64)

    for i in range(batch_size):
        nq = q_counts[i]
        nk = k_counts[i]
        if nq>0:
            coords_q[i,:nq,:] = coords_q_list[i]
            feats_q[i,:nq,:] = feats_q_list[i]
            feats_q_kv[i,:nq,:] = feats_q_list_kv[i]
            q_mask[i,:nq] = True
        if nk>0:
            coords_k[i,:nk,:] = coords_k_list[i]
            feats_k[i,:nk,:] = feats_k_list[i]
            k_mask[i,:nk] = True
            k_atom_types[i,:nk] = k_types_list[i]
        # pad edge_types_qk
        if nq>0 and nk>0:
            edge_types_qk[i,:nq,:nk] = edge_types_qk_list[i]

    return {
        'coords_q': torch.tensor(coords_q),
        'coords_k': torch.tensor(coords_k),
        'feats_q': torch.tensor(feats_q),
        'feats_q_kv': torch.tensor(feats_q_kv),
        'feats_k': torch.tensor(feats_k),
        'q_mask': torch.tensor(q_mask),
        'k_mask': torch.tensor(k_mask),
        'k_atom_types': torch.tensor(k_atom_types),
        'edge_types_qk': torch.tensor(edge_types_qk),
        'labels_num': torch.as_tensor(labels_nums, dtype=torch.int64),
        'labels_position': torch.as_tensor(np.array(labels_position), dtype=torch.int64),
        'labels_height': torch.as_tensor(np.array(labels_height), dtype=torch.int64),
        'q_counts': torch.tensor(q_counts),
        'k_counts': torch.tensor(k_counts)
    }


atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formalCharge = [-1, -2, 1, 2, 0]
degree = [0, 1, 2, 3, 4, 5, 6]
num_Hs = [0, 1, 2, 3, 4]
local_chiral_tags = [0, 1, 2, 3] 
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def getNodeFeatures(list_rdkit_atoms, owningMol, global_chirality=True):
    F_v = (len(atomTypes)+1) +\
        (len(degree)+1) + \
        (len(formalCharge)+1) +\
        (len(num_Hs)+1)+\
        (len(hybridization)+1) +\
        2 + 4 + 5 # 52
    
    if global_chirality:
        global_tags = dict(rdkit.Chem.FindMolChiralCenters(owningMol, force=True, includeUnassigned=True, useLegacyImplementation=False))
    else:
        global_tags = []
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes) # atom symbol, dim=12 + 1 
        features += one_hot_embedding(node.GetTotalDegree(), degree) # total number of bonds, H included, dim=7 + 1
        features += one_hot_embedding(node.GetFormalCharge(), formalCharge) # formal charge, dim=5+1 
        features += one_hot_embedding(node.GetTotalNumHs(), num_Hs) # total number of bonded hydrogens, dim=5 + 1
        features += one_hot_embedding(node.GetHybridization(), hybridization) # hybridization state, dim=7 + 1
        features += [int(node.GetIsAromatic())] # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass() * 0.01]  # atomic mass / 100, dim=1

        # Chiral tags (global and local)
        idx = node.GetIdx()
        global_chiral_tag = 0
        if idx in global_tags:
            if global_tags[idx] == 'R':
                global_chiral_tag = 1
            elif global_tags[idx] == 'S':
                global_chiral_tag = 2
            else:
                global_chiral_tag = -1
        
        features += one_hot_embedding(global_chiral_tag, [0, 1, 2])  # global chiral tag, dim=4
        features += one_hot_embedding(node.GetChiralTag(), local_chiral_tags)  # local chiral tag, dim=5
        
        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)


def get_args():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Min learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', help='type of scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use (cpu or cuda)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes (1 for regression, 2 for R/S, 3 for chiral type)')
    parser.add_argument('--random_ratio', type=float, default=0., help='Ratio of molecules with perturbed chiral centers (for ablation)')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing dataset files (default: PROJECT_ROOT/codes/data)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for saving results (default: PROJECT_ROOT/outputs)')
    # MoleculeNet dataset
    parser.add_argument('--dataset', type=str, default='sider', help='MoleculeNet dataset name', choices=[
        'bbbp', 'sider', 'clintox', 'freesolv', 'bace'
    ])

    # model structure
    parser.add_argument('--use_qr', action='store_true', help='Use QR orthogonalization in CDKernelVolume')
    parser.add_argument('--reg_lambda', type=float, default=1., help='Orthogonalization regularization weight')
    parser.add_argument('--use_orth_reg', action='store_true', help='Add orthogonalization regularization to loss')
    parser.add_argument('--chiral_encoder', type=str, default="Kernel", help='Chiral encoder type', choices=["Kernel", "Linear"])
    # model size
    parser.add_argument('--proj_dim', type=int, default=64, help='Project dimension size for the CDkernel')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lambda_mse', type=float, default=1., help='MSE loss weight for ranking task')
    parser.add_argument('--lambda_rank', type=float, default=1., help='Ranking loss weight for ranking task')

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = os.path.join(_PROJECT_ROOT, 'codes', 'data')
    if args.output_dir is None:
        args.output_dir = os.path.join(_PROJECT_ROOT, 'outputs')
    return args