"""
Central chirality enantiomer ranking task.
Ranks pairs of enantiomers by docking score (e.g., for virtual screening).
"""
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle
import pandas as pd
from rdkit import Chem
import random
from collections import defaultdict


def run_epoch(model, loader, device, loss_fn, optimizer=None,
              use_orth_reg=False, reg_lambda=1.0, mode="train",
              margin=0.3, lambda_mse=1., lambda_rank=1., save_path=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    test_res = [[], []] if mode == "test" else None
    
    criterion = torch.nn.MarginRankingLoss(margin=margin)

    for batch in tqdm(loader, desc=f"{mode.capitalize()} Batches", leave=False, disable=not sys.stdout.isatty()):
        feats_q = batch['feats_q'].to(device)
        feats_q_kv = batch['feats_q_kv'].to(device)
        feats_k = batch['feats_k'].to(device)
        coords_q = batch['coords_q'].to(device)
        coords_k = batch['coords_k'].to(device)
        q_mask = batch['q_mask'].to(device)
        k_mask = batch['k_mask'].to(device)
        k_types = batch['k_atom_types'].to(device)
        edge_types_qk = batch['edge_types_qk'].to(device)
        labels = batch['labels'].to(device).float()

        if mode == "train":
            optimizer.zero_grad()

        output, _, loss_orth_reg = model(feats_q, feats_q_kv, feats_k, k_types, edge_types_qk,
                                         coords_q, coords_k, q_mask, k_mask)

        loss = loss_fn(output, labels)

        # margin ranking loss
        rank_targets = torch.sign((labels[0::2] - labels[1::2]) + 1e-8).squeeze()
        loss_relative = criterion(output[0::2].squeeze(), 
                                  output[1::2].squeeze(), 
                                  rank_targets)
        loss = lambda_mse*loss + lambda_rank*loss_relative

        if use_orth_reg:
            loss = loss + reg_lambda * loss_orth_reg

        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

        total_loss += loss.item() * len(labels)

        target_ranking = ((torch.round(labels[0::2] * 100.) / 100.) > 
                          (torch.round(labels[1::2] * 100.) / 100.)).float()
        output_ranking = ((torch.round(output[0::2] * 100.) / 100.) > 
                          (torch.round(output[1::2] * 100.) / 100.)).float()
        total_acc += torch.sum(output_ranking == target_ranking)
        count += len(labels)

        if mode == "test":
            test_res[0].extend(labels.cpu().numpy().tolist())
            test_res[1].extend(output.detach().cpu().numpy().tolist())

    avg_loss = total_loss / count
    avg_acc = (total_acc / (count/2)).item()

    return avg_loss, avg_acc, test_res


def train_hct(train_data, val_data, test_data, args):
    # Dataset & Dataloader
    train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, collate_fn=collate_hct)
    val_loader = DataLoader(val_data, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, collate_fn=collate_hct)
    test_loader = DataLoader(test_data, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, collate_fn=collate_hct)

    model = HCTModel(d_model=args.hidden_dim, n_heads=args.num_heads, num_layers=args.num_layers, proj_dim=args.proj_dim, 
                     chiral_encoder=args.chiral_encoder, use_qr=args.use_qr, ecd=False).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')  # val_
    best_test = 0
    best_dict = {}
    for epoch in range(args.epochs):
        train_loss, train_acc, _ = run_epoch(model, train_loader, args.device, loss_fn, optimizer, 
                                             use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="train", 
                                             lambda_mse=args.lambda_mse, lambda_rank=args.lambda_rank)
        val_loss, val_acc, _ = run_epoch(model, val_loader, args.device, loss_fn, 
                                         use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="val", 
                                         lambda_mse=args.lambda_mse, lambda_rank=args.lambda_rank)
        test_loss, test_acc, test_res = run_epoch(model, test_loader, args.device, loss_fn, 
                                                  use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="test", 
                                                  lambda_mse=args.lambda_mse, lambda_rank=args.lambda_rank)
        scheduler.step()
        # save the best loss via val_loss and test_res
        if test_acc > best_test:
            best_test = test_acc
        if val_loss < best_loss:
            best_loss = val_loss
            best_dict = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_labels": test_res[0],
            "test_preds": test_res[1],
            "best_test_acc": best_test
        }

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = f"ranking_{args.epochs}_{args.bs}_{args.lr}.pkl"
    with open(os.path.join(args.output_dir, save_name), "wb") as f:
        best_dict["best_test_acc"] = best_test
        pickle.dump(best_dict, f)

    return model


if __name__ == '__main__':
    args = get_args()
    from data_processing import process_central_ranking

    train_path = os.path.join(args.data_dir, "ranking_train.pkl")
    val_path = os.path.join(args.data_dir, "ranking_validation.pkl")
    test_path = os.path.join(args.data_dir, "ranking_test.pkl")
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Ranking data not found. Place train/val/test pkl files in {args.data_dir}. "
            "See README for data sources (ChIRo enantiomer ranking splits)."
        )
    train_df = pd.read_pickle(train_path)
    val_df = pd.read_pickle(val_path)
    test_df = pd.read_pickle(test_path)

    train_data = process_central_ranking(train_df, disable_tqdm=not sys.stdout.isatty())
    val_data = process_central_ranking(val_df, disable_tqdm=not sys.stdout.isatty())
    test_data = process_central_ranking(test_df, disable_tqdm=not sys.stdout.isatty())

    def sample_pairs(df, data_dict, seed=42):
        random.seed(seed)
        groups = defaultdict(lambda: {"a": [], "b": []})
        for idx, mol_id in zip(df.index, df["ID"]):
            base_id = mol_id.replace("@", "")
            if "@@" in mol_id:
                groups[base_id]["b"].append(idx)
            else:
                groups[base_id]["a"].append(idx)
        pairs = []
        for base_id, group in groups.items():
            a_list, b_list = group["a"], group["b"]
            if not a_list or not b_list:
                continue
            a_idx = random.choice(a_list)
            b_idx = random.choice(b_list)
            pairs.append((a_idx, b_idx))
        random.shuffle(pairs)
        sampled_data = []
        for a, b in pairs:
            mol = df["rdkit_mol_cistrans_stereo"][a]
            atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
            node_features = getNodeFeatures(atoms, mol)
            data_dict[a]["atom_onehot"] = node_features
            data_dict[a]['rdkit_mol'] = mol
            sampled_data.append(data_dict[a])
            mol = df["rdkit_mol_cistrans_stereo"][b]
            atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
            node_features = getNodeFeatures(atoms, mol)
            data_dict[b]["atom_onehot"] = node_features
            data_dict[b]['rdkit_mol'] = mol
            sampled_data.append(data_dict[b])
        return sampled_data

    train_data = sample_pairs(train_df, train_data)
    val_data = sample_pairs(val_df, val_data)
    test_data = sample_pairs(test_df, test_data)

    model = train_hct(
        train_data, val_data, test_data, args
        )
