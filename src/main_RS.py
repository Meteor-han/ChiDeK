"""
Central chirality R/S configuration classification task.
Binary classification of R vs S configuration for chiral molecules.
"""
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle
import pandas as pd
import random


def run_epoch(model, loader, device, loss_fn, optimizer=None, 
              use_orth_reg=False, reg_lambda=1.0, mode="train", save_path=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    test_res = [[], []] if mode == "test" else None

    for batch in tqdm(loader, desc=f"{mode.capitalize()} Batches", leave=False):
        feats_q = batch['feats_q'].to(device)
        feats_q_kv = batch['feats_q_kv'].to(device)
        feats_k = batch['feats_k'].to(device)
        coords_q = batch['coords_q'].to(device)
        coords_k = batch['coords_k'].to(device)
        q_mask = batch['q_mask'].to(device)
        k_mask = batch['k_mask'].to(device)
        k_types = batch['k_atom_types'].to(device)
        edge_types_qk = batch['edge_types_qk'].to(device)
        labels = batch['labels'].to(device)

        if mode == "train":
            optimizer.zero_grad()

        output, _, loss_orth_reg = model(feats_q, feats_q_kv, feats_k, k_types, edge_types_qk,
                                         coords_q, coords_k, q_mask, k_mask)
        loss = loss_fn(output, labels.long())
        if use_orth_reg:
            loss = loss + reg_lambda * loss_orth_reg
        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        total_acc += compute_accuracy_multiclasses(output, labels) * len(labels)
        count += len(labels)

        if mode == "test":
            preds = torch.argmax(output, dim=1)
            # preds = (torch.sigmoid(output) > 0.5).float()
            test_res[0].extend(labels.cpu().numpy().tolist())
            test_res[1].extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / count
    avg_acc = total_acc / count

    return avg_loss, avg_acc, test_res


def train_hct(train_data, val_data, test_data, args):
    device=args.device
    epochs=args.epochs
    batch_size=args.bs
    lr=args.lr
    weight_decay=args.weight_decay
    use_qr=args.use_qr
    reg_lambda=args.reg_lambda
    use_orth_reg=args.use_orth_reg
    # Dataset & Dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_hct)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_hct)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_hct)

    model = HCTModel(d_model=args.hidden_dim, n_heads=args.num_heads, num_layers=args.num_layers, proj_dim=args.proj_dim, 
                     chiral_encoder=args.chiral_encoder, use_qr=args.use_qr, ecd=False, num_classes=args.num_classes).to(args.device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = [None, float('inf'), None]  # train_loss, val_loss, test_res
    for epoch in range(epochs):
        train_loss, train_acc, _ = run_epoch(model, train_loader, device, loss_fn,
                                        optimizer, use_orth_reg=use_orth_reg, reg_lambda=reg_lambda, mode="train")
        val_loss, val_acc, _ = run_epoch(model, val_loader, device, loss_fn,
                                    use_orth_reg=use_orth_reg, reg_lambda=reg_lambda, mode="val")
        test_loss, test_acc, test_res = run_epoch(model, test_loader, device, loss_fn,
                                        use_orth_reg=use_orth_reg, reg_lambda=reg_lambda, mode="test")
        if val_loss < best_loss[1]:
            best_loss = [train_loss, val_loss, test_res]
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "RS_test_res.pkl"), "wb") as f:
                pickle.dump({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_labels": test_res[0],
                    "test_preds": test_res[1]
                }, f)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    return model


if __name__ == '__main__':
    args = get_args()
    from data_processing import process_central_RS

    train_path = os.path.join(args.data_dir, "RS_train.pkl")
    val_path = os.path.join(args.data_dir, "RS_validation.pkl")
    test_path = os.path.join(args.data_dir, "RS_test.pkl")
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"R/S classification data not found. Place train/val/test pkl files in {args.data_dir}. "
            "See README for data preparation."
        )
    train_df = pd.read_pickle(train_path)
    val_df = pd.read_pickle(val_path)
    test_df = pd.read_pickle(test_path)

    train_data = process_central_RS(train_df, "train", args.random_ratio, args.data_dir)
    val_data = process_central_RS(val_df, "val", args.random_ratio, args.data_dir)
    test_data = process_central_RS(test_df, "test", args.random_ratio, args.data_dir)

    args.num_classes = 2
    model = train_hct(
        train_data, val_data, test_data, args
        )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(args.output_dir, "RS_ckpt.pth"))
