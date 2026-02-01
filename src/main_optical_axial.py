"""
Axial chirality optical rotation (OR) prediction task.
Binary classification of OR sign at 589nm for axially chiral molecules.
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
        labels = batch['labels'].to(device).long()

        if mode == "train":
            optimizer.zero_grad()

        output, _, loss_orth_reg = model(feats_q, feats_q_kv, feats_k, k_types, edge_types_qk,
                                         coords_q, coords_k, q_mask, k_mask)
        loss = loss_fn(output, labels)
        if use_orth_reg:
            loss = loss + reg_lambda * loss_orth_reg
        if mode == "train":
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            for name, p in model.named_parameters():
                try:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print("NaN in grad:", name)
                except:
                    print("Error checking grad:", name)
                    pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        total_acc += compute_accuracy_multiclasses(output, labels) * len(labels)
        count += len(labels)

        if mode == "test":
            preds = torch.argmax(output, dim=1)
            test_res[0].extend(labels.cpu().numpy().tolist())
            test_res[1].extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / count
    avg_acc = total_acc / count

    return avg_loss, avg_acc, test_res


def train_hct(train_data, val_data, test_data, args):
    # Dataset & Dataloader
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, collate_fn=collate_hct)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False, collate_fn=collate_hct)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False, collate_fn=collate_hct)

    model = HCTModel(d_model=args.hidden_dim, n_heads=args.num_heads, num_layers=args.num_layers, proj_dim=args.proj_dim, 
                     chiral_encoder=args.chiral_encoder, use_qr=args.use_qr, ecd=False, num_classes=args.num_classes).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = [None, float('inf'), None]  # train_loss, val_loss, test_res
    for epoch in range(args.epochs):
        train_loss, train_acc, _ = run_epoch(model, train_loader, args.device, loss_fn,
                                        optimizer, use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="train")
        val_loss, val_acc, _ = run_epoch(model, val_loader, args.device, loss_fn,
                                    use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="val")
        test_loss, test_acc, test_res = run_epoch(model, test_loader, args.device, loss_fn,
                                        use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="test")
        # save the best loss via val_loss and test_res
        if val_loss < best_loss[1]:
            best_loss = [train_loss, val_loss, test_res]
            os.makedirs(args.output_dir, exist_ok=True)
            prefix = f"optical_axial_ep{args.epochs}_{args.bs}_{args.lr}"
            with open(os.path.join(args.output_dir, f"{prefix}_test_res.pkl"), "wb") as f:
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
    from data_processing import process_axial_chirality

    data_path = os.path.join(args.data_dir, "hct_ecd_axial.pkl")
    data = pd.read_pickle(data_path)
    data = process_axial_chirality(data, args.random_ratio, args.data_dir, label_mode="optical", disable_tqdm=not sys.stdout.isatty())

    split_path = os.path.join(args.data_dir, "ecd_axial_index_split.pkl")
    with open(split_path, "rb") as f:
        index_ = pickle.load(f)
        train_index, val_index, test_index = index_["train_index"], index_["val_index"], index_["test_index"]
    
    train_data = [data[i] for i in train_index]
    val_data = [data[i] for i in val_index]
    test_data = [data[i] for i in test_index]

    args.num_classes = 2
    model = train_hct(
        train_data, val_data, test_data, args
        )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(args.output_dir, "optical_axial_ckpt.pth"))
