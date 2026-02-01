"""
Axial chirality ECD (Electronic Circular Dichroism) spectrum prediction task.
Predicts peak number, position, and height for axially chiral molecules.
"""
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle
import pandas as pd
import random


def run_epoch(model, loader, device, optimizer=None,
              use_orth_reg=False, reg_lambda=1.0, mode="train",
              margin=0.3, save_path=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    count = 0
    loss_accum = 0
    loss_accum_3 = [0, 0, 0]  # for number, position, height
    test_res = {"num": [[], []], "position": [[], []], "height": [[], []]} if mode=="test" else None # labels, preds
    total_number = []
    total_position = []
    total_height = []

    ce_loss = torch.nn.CrossEntropyLoss()

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
        num_gt = batch['labels_num'].to(device)
        pos_gt = batch['labels_position'].to(device)
        height_gt = batch['labels_height'].to(device)

        num_pred, pos_pred, height_pred, _, loss_orth_reg = model(
            feats_q, feats_q_kv, feats_k, k_types, edge_types_qk,
            coords_q, coords_k, q_mask, k_mask
        )
        pos_pred = pos_pred.view(-1, 7, 20)  # [batch, 7, 20]
        height_pred = height_pred.view(-1, 7, 2)  # [batch, 7, 2]
        # transform the groundtruth and prediction labels
        new_gt_pos, new_pred_pos = [], []
        new_gt_height, new_pred_height = [], []
        for i in range(num_gt.size(0)):
            if num_gt[i] == 0:
                # just give the same one num
                new_gt_pos.append(torch.zeros((7,), dtype=torch.int64).to(device))
                temp_ = torch.zeros((7, 20)).to(device)
                temp_[:, 0] = 1.0
                new_pred_pos.append(temp_)
                new_gt_height.append(torch.zeros((7,), dtype=torch.int64).to(device))
                temp_ = torch.zeros((7, 2)).to(device)
                temp_[:, 0] = 1.0
                new_pred_height.append(temp_)
            else:
                new_gt_pos.append(pos_gt[i, :int(num_gt[i])])
                new_pred_pos.append(pos_pred[i, :int(num_gt[i]), :])
                new_gt_height.append(height_gt[i, :int(num_gt[i])])
                new_pred_height.append(height_pred[i, :int(num_gt[i]), :])
        
        new_gt_pos_tensor = torch.cat(new_gt_pos, dim=0)            # [batch*node_num]
        new_pred_pos_tensor = torch.cat(new_pred_pos, dim=0)        # [batch*node_num, 20]
        new_gt_height_tensor = torch.cat(new_gt_height, dim=0)      # [batch*node_num]
        new_pred_height_tensor = torch.cat(new_pred_height, dim=0)    # [batch*node_num, 2]
        assert new_gt_pos_tensor.max().item() < 20
        assert new_gt_height_tensor.max().item() < 2

        # backward propagation
        loss_pos = ce_loss(new_pred_pos_tensor, new_gt_pos_tensor)
        loss_height = ce_loss(new_pred_height_tensor, new_gt_height_tensor)
        loss_num = ce_loss(num_pred, num_gt)
        loss = loss_num + 2*loss_height + loss_pos

        if use_orth_reg:
            loss = loss + reg_lambda * loss_orth_reg
        if mode == "train":
            optimizer.zero_grad()
        if mode == "train":
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            for name, p in model.named_parameters():
                try:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print("NaN in grad:", name)
                except:
                    pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

        loss_accum += loss.detach().cpu().item() * num_gt.size(0)
        # if nan, then debug
        if np.isnan(loss_accum):
            print("Error: loss is nan!")
        count += num_gt.size(0)
        loss_accum_3[0] += loss_num.detach().cpu().item() * num_gt.size(0)
        loss_accum_3[1] += loss_pos.detach().cpu().item() * num_gt.size(0)
        loss_accum_3[2] += loss_height.detach().cpu().item() * num_gt.size(0)

        # number rmse
        _, num_preds = torch.max(num_pred, dim=1)
        num_rmse = torch.sqrt(torch.mean((num_preds - num_gt).float() ** 2)).detach().cpu().item()
        total_number.append(num_rmse)
        _, pos_preds = torch.max(new_pred_pos_tensor, dim=1)
        pos_rmse_list = []
        pos_offset = 0
        for i in range(num_gt.size(0)):
            gt_num = int(num_gt[i].item())
            _, pred_num = torch.max(num_pred[i].unsqueeze(0), dim=1)
            pred_num = int(pred_num.item())
            n = min(gt_num, pred_num)
            if n == 0:
                pos_rmse_list.append(torch.as_tensor(0.0).item())
                continue
            gt_pos = new_gt_pos_tensor[pos_offset:pos_offset + gt_num][:n]
            pred_pos = pos_preds[pos_offset:pos_offset + gt_num][:n]
            pos_rmse_list.append(torch.sqrt(torch.mean((pred_pos - gt_pos).float() ** 2)).item())
            pos_offset += gt_num
        pos_rmse = np.mean(pos_rmse_list) if pos_rmse_list else 0.0
        total_position.append(pos_rmse)

        _, height_preds = torch.max(new_pred_height_tensor, dim=1)
        total_height.extend((height_preds == new_gt_height_tensor).detach().cpu().numpy().tolist())

        if mode == "test":
            # save the prediction results
            test_res["num"][0].append(num_gt.cpu().numpy().tolist())
            test_res["num"][1].append(num_preds.detach().cpu().numpy().tolist())
            test_res["position"][0].append(new_gt_pos_tensor.cpu().numpy().tolist())
            test_res["position"][1].append(pos_preds.detach().cpu().numpy().tolist())
            test_res["height"][0].append(new_gt_height)
            test_res["height"][1].append(new_pred_height)

    final_num_rmse = np.mean(total_number) if total_number else 0.0
    final_pos_rmse = np.mean(total_position) if total_position else 0.0
    final_height_acc = np.sum(total_height) / len(total_height) if total_height else 0.0

    return loss_accum / count, loss_accum_3[0] / count, loss_accum_3[1] / count, loss_accum_3[2] / count, final_num_rmse, final_pos_rmse, final_height_acc, test_res


def train_hct(train_data, val_data, test_data, args):
    # Dataset & Dataloader
    train_loader = DataLoader(train_data, batch_size=args.bs, num_workers=args.num_workers, shuffle=True, collate_fn=collate_hct_ecd)
    val_loader = DataLoader(val_data, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, collate_fn=collate_hct_ecd)
    test_loader = DataLoader(test_data, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, collate_fn=collate_hct_ecd)
        
    model = HCTModel(d_model=args.hidden_dim, n_heads=args.num_heads, num_layers=args.num_layers, proj_dim=args.proj_dim, 
                     chiral_encoder=args.chiral_encoder, use_qr=args.use_qr, ecd=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_acc = float('-inf')  # val_
    best_dict = {}
    for epoch in tqdm(range(args.epochs)):#, disable=not sys.stdout.isatty()):
        train_loss, train_loss_num, train_loss_pos, train_loss_height, train_num_rmse, train_pos_rmse, train_height_acc, _ = run_epoch(model, train_loader, args.device,
                                        optimizer, use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="train")
        val_loss, val_loss_num, val_loss_pos, val_loss_height, val_num_rmse, val_pos_rmse, val_height_acc, _ = run_epoch(model, val_loader, args.device,
                                    use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="val")
        test_loss, test_loss_num, test_loss_pos, test_loss_height, test_num_rmse, test_pos_rmse, test_height_acc, test_res = run_epoch(model, test_loader, args.device,
                                        use_orth_reg=args.use_orth_reg, reg_lambda=args.reg_lambda, mode="test")
        scheduler.step()
        if val_height_acc > best_acc:
            best_acc = val_height_acc
            best_dict = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_loss_num": train_loss_num,
            "train_loss_pos": train_loss_pos,
            "train_loss_height": train_loss_height,
            "train_num_rmse": train_num_rmse,
            "train_pos_rmse": train_pos_rmse,
            "train_height_acc": train_height_acc,
            "val_loss": val_loss,
            "val_loss_num": val_loss_num,
            "val_loss_pos": val_loss_pos,
            "val_loss_height": val_loss_height,
            "val_num_rmse": val_num_rmse,
            "val_pos_rmse": val_pos_rmse,
            "val_height_acc": val_height_acc,
            "test_loss": test_loss,
            "test_loss_num": test_loss_num,
            "test_loss_pos": test_loss_pos,
            "test_loss_height": test_loss_height,
            "test_num_rmse": test_num_rmse,
            "test_pos_rmse": test_pos_rmse,
            "test_height_acc": test_height_acc,
            "test_res": test_res
        }

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Train Loss Num: {train_loss_num:.4f}, Pos: {train_loss_pos:.4f}, Height: {train_loss_height:.4f} | ")
        print(f"Val Loss Num: {val_loss_num:.4f}, Pos: {val_loss_pos:.4f}, Height: {val_loss_height:.4f} | ")
        print(f"Test Loss Num: {test_loss_num:.4f}, Pos: {test_loss_pos:.4f}, Height: {test_loss_height:.4f} | ")
        print(f"Train Num RMSE: {train_num_rmse:.4f}, Pos RMSE: {train_pos_rmse:.4f}, Height Acc: {train_height_acc:.4f} | ")
        print(f"Val Num RMSE: {val_num_rmse:.4f}, Pos RMSE: {val_pos_rmse:.4f}, Height Acc: {val_height_acc:.4f} | ")
        print(f"Test Num RMSE: {test_num_rmse:.4f}, Pos RMSE: {test_pos_rmse:.4f}, Height Acc: {test_height_acc:.4f} | ")

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = f"ecd_axial_{args.random_ratio}_{args.epochs}_{args.bs}_{args.lr}.pkl"
    with open(os.path.join(args.output_dir, save_name), "wb") as f:
        pickle.dump(best_dict, f)

    return model


if __name__ == '__main__':
    args = get_args()
    from data_processing import process_axial_chirality

    data_path = os.path.join(args.data_dir, "hct_ecd_axial.pkl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. See README for data preparation.")
    data = pd.read_pickle(data_path)
    data = process_axial_chirality(data, args.random_ratio, args.data_dir, label_mode="ecd", disable_tqdm=not sys.stdout.isatty())

    split_path = os.path.join(args.data_dir, "ecd_axial_index_split.pkl")
    with open(split_path, "rb") as f:
        index_ = pickle.load(f)
        train_index, val_index, test_index = index_["train_index"], index_["val_index"], index_["test_index"]
    
    train_data = [data[i] for i in train_index]
    val_data = [data[i] for i in val_index]
    test_data = [data[i] for i in test_index]

    model = train_hct(train_data, val_data, test_data, args)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(args.output_dir, "ecd_axial_ckpt.pth"))