import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


@torch.jit.script
def gaussian(x, mean, std):
    """
    Gaussian function implemented for PyTorch tensors.

    :param x: The input tensor.
    :param mean: The mean for the Gaussian function.
    :param std: The standard deviation for the Gaussian function.

    :return: The output tensor after applying the Gaussian function.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """
    A neural network module implementing a Gaussian layer, useful in graph neural networks.

    Attributes:
        - K: Number of Gaussian kernels.
        - means, stds: Embeddings for the means and standard deviations of the Gaussian kernels.
        - mul, bias: Embeddings for scaling and bias parameters.
    """
    def __init__(self, K=128, edge_types=1024):
        """
        Initializes the GaussianLayer module.

        :param K: Number of Gaussian kernels.
        :param edge_types: Number of different edge types to consider.

        :return: An instance of the configured Gaussian kernel and edge types.
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        Forward pass of the GaussianLayer.

        :param x: Input tensor representing distances or other features.
        :param edge_type: Tensor indicating types of edges in the graph.

        :return: Tensor transformed by the Gaussian layer.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


def get_activation_fn(activation):
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "leaky_relu":
        return F.leaky_relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))
    

class NonLinearHead(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network 
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """
    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        """
        Initializes the NonLinearHead module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        :param hidden: Dimension of the hidden layer; defaults to the same as input_dim if not provided.
        """
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x
    

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    """
    Manual Multi-head cross-attention to allow:
     - different linear projections for keys/values depending on atom subtype
     - masking (key padding mask) and query padding mask handled externally

    Inputs:
      q: (B, Q, d_model)
      k: (B, K, d_model)
      v: (B, K, d_model)
      k_types: (B, K) ints used to choose which linear to apply for keys/vals
      q_mask: (B, Q) bool True for valid
      k_mask: (B, K) bool True for valid
      coords_q: (B, Q, 3)
      coords_k: (B, K, 3)
    Outputs:
      updated_q: (B, Q, d_model)
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # query projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # keys/values: we will have two separate linear layers for type 1 (chiral_related) and type 2 (non_related)
        self.k_proj_type1 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_type1 = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_type2 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_type2 = nn.Linear(d_model, d_model, bias=False)

        # output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, k_types, q_mask, k_mask, attn_bias=None):
        """
        q: (B, Q, d)
        k: (B, K, d)
        v: (B, K, d)
        k_types: (B, K) values {1,2}
        q_mask: (B, Q) bool
        k_mask: (B, K) bool
        coords_q: (B, Q, 3)
        coords_k: (B, K, 3)
        """
        B, Q, d = q.shape
        _, K, _ = k.shape
        device = q.device

        # project queries
        q_proj = self.q_proj(q)  # (B, Q, d)
        q_heads = q_proj.view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)  # (B, heads, Q, head_dim)

        # Project keys/values per atom type (chiral_related vs non_related)
        k1 = self.k_proj_type1(k)  # (B, K, D)
        v1 = self.v_proj_type1(v)
        k2 = self.k_proj_type2(k)
        v2 = self.v_proj_type2(v)

        mask1 = (k_types == 1).unsqueeze(-1)   # (B, K, 1)
        mask2 = (k_types == 2).unsqueeze(-1)

        k_proj_all = torch.where(mask1, k1,
                        torch.where(mask2, k2, torch.zeros_like(k1)))
        v_proj_all = torch.where(mask1, v1,
                        torch.where(mask2, v2, torch.zeros_like(v1)))

        # shape for heads
        k_heads = k_proj_all.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)  # (B, heads, K, head_dim)
        v_heads = v_proj_all.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)  # (B, heads, K, head_dim)

        # compute attention logits: (B, heads, Q, K)
        # scaled dot product
        logits = torch.matmul(q_heads, k_heads.transpose(-2, -1))  # (B, heads, Q, K)
        logits = logits / math.sqrt(self.head_dim)

        if attn_bias is not None:
            logits = logits + attn_bias  # add learned bias

        # mask keys: k_mask (B, K) -> expand to (B, 1, 1, K) or (B, heads, Q, K)
        # True indicates valid, we want to set logits to -inf where invalid
        k_mask_expand = k_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,K)
        logits = logits.masked_fill(~k_mask_expand, float('-1e9'))

        # Padded query positions are zeroed in output via q_mask
        attn = F.softmax(logits, dim=-1)  # (B, heads, Q, K)
        attn = self.dropout(attn)

        out_heads = torch.matmul(attn, v_heads)  # (B, heads, Q, head_dim)
        out = out_heads.transpose(1, 2).contiguous().view(B, Q, d)  # (B, Q, d)
        out = self.out_proj(out)  # (B, Q, d)
        return out, logits, attn


class HCTLayer(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, k_types, q_mask, k_mask, attn_bias=None):
        # q: (B, Q, d)
        # cross-attn updates queries only
        q_res = q
        attn_out, attn_bias, attn_weights = self.cross_attn(
            q, k, v, k_types, q_mask, k_mask, attn_bias=attn_bias
        )
        q = self.norm1(q_res + attn_out)
        q_res2 = q
        q = self.norm2(q_res2 + self.ffn(q_res2))
        # optionally zero out padded query rows
        q = q * q_mask.unsqueeze(-1).type_as(q)
        return q, attn_bias, attn_weights


class HCTModel(nn.Module):
    def __init__(self, num_atom_types=52, 
                 d_model=256, n_heads=8, num_layers=4, mlp_ratio=4.0, dropout=0.1, num_rbfs=128, 
                 proj_dim=64, chiral_encoder="Kernel",
                 num_peaks=7, num_pos=20,
                 use_qr=False, ecd=False, num_classes=1):
        super().__init__()
        self.ecd = ecd
        self.d_model = d_model
        self.dist_cutoff = 20. # could be large?
        self.n_heads = n_heads
        self.num_rbfs = num_rbfs
        self.num_classes = num_classes

        self.gbf = GaussianLayer(num_rbfs, edge_types=3)
        self.rbf_nonlinear = NonLinearHead(num_rbfs, n_heads, "gelu")

        self.q_embed = CDKernelVolume(proj_dim=proj_dim, chiral_encoder=chiral_encoder, K=d_model, use_qr=use_qr, ecd=ecd)  # for chiral centers (queries)
        # add a learnable chiral token to queries
        self.chiral_token = nn.Parameter(torch.randn(1, 1, d_model))
        # initialize token
        nn.init.xavier_uniform_(self.chiral_token)

        self.q_embed_kv = NonLinearHead(num_atom_types, d_model, "gelu")  # chiral
        self.k_embed_type1 = NonLinearHead(num_atom_types, d_model, "gelu")  # chiral_related
        self.k_embed_type2 = NonLinearHead(num_atom_types, d_model, "gelu")  # non_related
        # self.qnorm = nn.LayerNorm(d_model)

        # stack L HCT layers
        self.layers = nn.ModuleList([
            HCTLayer(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

        # final pooling / head
        self.pool_norm = nn.LayerNorm(d_model)
        if ecd:
            self.chiral = NonLinearHead(d_model, d_model, "gelu")
            self.non_chiral = NonLinearHead(d_model, d_model, "gelu")
            # prediction heads
            self.pred_number_layer = ClassificationHead(
                input_dim=d_model,
                inner_dim=d_model,
                num_classes=num_peaks,   # 0-6, 0-8
                activation_fn="gelu",
                pooler_dropout=0.1,
            )
            # [B, emb] to [B, 7, 20]

            self.pred_position_layer = ClassificationHead(
                input_dim=d_model,
                inner_dim=d_model,
                num_classes=num_peaks*num_pos,  # 20
                activation_fn="gelu",
                pooler_dropout=0.1,
            )
            self.pred_height_layer = ClassificationHead(
                input_dim=d_model,
                inner_dim=d_model,
                num_classes=num_peaks*2,
                activation_fn="gelu",
                pooler_dropout=0.1,
            )
        else:
            self.head = ClassificationHead(
                input_dim=d_model,
                inner_dim=d_model,
                num_classes=self.num_classes,
                activation_fn="gelu",
                pooler_dropout=0.1,
            )

    def forward(self, feats_q, feats_q_kv, feats_k, k_types, edge_types_qk, coords_q, coords_k, q_mask, k_mask):
        """
        feats_q: (B, Q, F)
        feats_k: (B, K, F)
        k_types: (B, K) ints {1,2}
        coords_q: (B, Q, 3)
        coords_k: (B, K, 3)
        q_mask: (B, Q) bool
        k_mask: (B, K) bool
        """
        B, Q, F = feats_q.shape
        _, K, _ = feats_k.shape
        device = feats_q.device

        # compute distance bias per head
        # compute pairwise distances between coords_q and coords_k (per batch)
        # coords_q: (B, Q, 3), coords_k: (B, K, 3)
        # produce (B, Q, K)
        coords_q_exp = coords_q.unsqueeze(2)  # (B, Q, 1, 3)
        # add the chiral token, expand coords_q as (0, 0, 0)
        chiral_token_coords = torch.zeros((B, 1, 1, 3), dtype=coords_q.dtype, device=device)
        coords_q_exp = torch.cat([chiral_token_coords, coords_q_exp], dim=1)  # (B, Q+1, 1, 3)

        coords_k_exp = coords_k.unsqueeze(1)  # (B, 1, K, 3)
        dists = torch.norm(coords_q_exp - coords_k_exp, dim=-1)  # (B, Q+1, K)
        # optionally clamp
        dists = torch.clamp(dists, 0.0, self.dist_cutoff)
        # expand edge type as 1
        Q += 1
        edge_types_qk = torch.cat([torch.ones((B,1,K), dtype=edge_types_qk.dtype, device=device), edge_types_qk], dim=1)  # (B, Q+1, K)

        # RBF -> (..., num_rbfs)
        rbf = self.gbf(dists, edge_types_qk)  # (B, Q, K, num_rbfs)
        # collapse B,Q,K by flatten then linear on last dim
        # apply rbf_linear to last dim to get per-head bias scalar (B, Q, K, heads)
        # We'll compute bias_heads shape (B, heads, Q, K)
        rbf_flat = rbf.view(B * Q * K, self.num_rbfs)
        bias_flat = self.rbf_nonlinear(rbf_flat)  # (B*Q*K, heads)
        attn_bias = bias_flat.view(B, Q, K, self.n_heads).permute(0, 3, 1, 2)  # (B, heads, Q, K)

        if not self.ecd:
            # embed queries
            q, loss_orth_reg = self.q_embed(feats_q, q_mask)  # (B, Q, d)
            q = q + self.q_embed_kv(feats_q_kv)

            # add the chiral token, expand the q_mask
            chiral_token_exp = self.chiral_token.expand(B, -1, -1)  # (B, 1, d)
            q = torch.cat([chiral_token_exp, q], dim=1)  # (B, Q+1, d)
            q_mask = torch.cat([torch.ones((B,1), dtype=torch.bool, device=device), q_mask], dim=1)  # (B, Q+1)

            k1 = self.k_embed_type1(feats_k)   # (B, K, d_model)
            k2 = self.k_embed_type2(feats_k)

            k = torch.where(k_types.unsqueeze(-1) == 1, k1,
                            torch.where(k_types.unsqueeze(-1) == 2, k2,
                                        torch.zeros_like(k1)))

            # values use same base as k (the MultiHeadCrossAttention will apply its own v_proj_* later)
            v = k.clone()

            # ensure padded positions are zeroed
            q = q * q_mask.unsqueeze(-1).type_as(q)
            k = k * k_mask.unsqueeze(-1).type_as(k)
            v = v * k_mask.unsqueeze(-1).type_as(v)

            # pass through L layers
            for layer in self.layers:
                q,  attn_bias, _ = layer(q, k, v, k_types, q_mask, k_mask, attn_bias)

            # pooling over queries for molecule representation (ignore padded queries)
            # q_mask True for real entries -> convert to float and compute masked mean
            q_mask_f = q_mask.type_as(q)  # (B, Q)
            denom = q_mask_f.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,1)
            mol_repr = (q * q_mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,d)
            mol_repr = self.pool_norm(mol_repr)

            out = self.head(mol_repr).squeeze(-1)  # (B,)
            return out, mol_repr, loss_orth_reg
        else:
            # embed queries
            q, q_non_chiral, loss_orth_reg = self.q_embed(feats_q, q_mask)  # (B, Q, d)
            q = q + self.q_embed_kv(feats_q_kv)

            q_non_chiral = 0.5 * (self.chiral(q) + self.non_chiral(q_non_chiral))
            q_non_chiral = q_non_chiral + self.q_embed_kv(feats_q_kv)

            # add the chiral token, expand the q_mask
            chiral_token_exp = self.chiral_token.expand(B, -1, -1)  # (B, 1, d)
            q = torch.cat([chiral_token_exp, q], dim=1)  # (B, Q+1, d)
            q_mask = torch.cat([torch.ones((B,1), dtype=torch.bool, device=device), q_mask], dim=1)  # (B, Q+1)
            # add the chiral token, expand the q_mask
            chiral_token_exp = self.chiral_token.expand(B, -1, -1)  # (B, 1, d)
            q_non_chiral = torch.cat([chiral_token_exp, q_non_chiral], dim=1)  # (B, Q+1, d)

            k1 = self.k_embed_type1(feats_k)   # (B, K, d_model)
            k2 = self.k_embed_type2(feats_k)

            k = torch.where(k_types.unsqueeze(-1) == 1, k1,
                            torch.where(k_types.unsqueeze(-1) == 2, k2,
                                        torch.zeros_like(k1)))
                
            # values use same base as k (the MultiHeadCrossAttention will apply its own v_proj_* later)
            v = k.clone()

            # ensure padded positions are zeroed
            q = q * q_mask.unsqueeze(-1).type_as(q)
            q_non_chiral = q_non_chiral * q_mask.unsqueeze(-1).type_as(q_non_chiral)
            k = k * k_mask.unsqueeze(-1).type_as(k)
            v = v * k_mask.unsqueeze(-1).type_as(v)

            # pass through L layers
            for layer in self.layers:
                q, attn_bias, _ = layer(q, k, v, k_types, q_mask, k_mask, attn_bias)
                q_non_chiral, attn_bias, _ = layer(q_non_chiral, k, v, k_types, q_mask, k_mask, attn_bias)

            # pooling over queries for molecule representation (ignore padded queries)
            # q_mask True for real entries -> convert to float and compute masked mean
            q_mask_f = q_mask.type_as(q)  # (B, Q)
            denom = q_mask_f.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B,1)
            mol_repr = (q * q_mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,d)
            mol_repr = self.pool_norm(mol_repr)
            # also get non-chiral mol_repr
            mol_repr_non_chiral = (q_non_chiral * q_mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,d)
            mol_repr_non_chiral = self.pool_norm(mol_repr_non_chiral)

            out_number = self.pred_number_layer(mol_repr_non_chiral).squeeze(-1)  # (B, num_classes)
            out_position = self.pred_position_layer(mol_repr_non_chiral).squeeze(-1)  # (B, num_classes)
            out_height = self.pred_height_layer(mol_repr).squeeze(-1)  # (B, num_classes)
            return out_number, out_position, out_height, (mol_repr, mol_repr_non_chiral), loss_orth_reg


class CDKernelVolume(nn.Module):
    """
    CDkernel embedding layer for wedge product volume.
    input: (x, y, z)
    output: [B, Q, K] embedding, orthogonalization regularization loss
    """
    def __init__(self, in_dim=3, proj_dim=64, chiral_encoder="Kernel", K=128, use_qr=False, ecd=False):
        super().__init__()
        self.ecd = ecd
        self.K = K
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.use_qr = use_qr
        self.chiral_encoder = chiral_encoder
        if chiral_encoder == "Kernel":
            self.weight = nn.Parameter(torch.randn(K, in_dim, proj_dim))
            nn.init.uniform_(self.weight, 0, 1)
        else:
            self.weight = nn.Linear(in_features=9, out_features=K)
            nn.init.uniform_(self.weight.weight, 0, 1)
            nn.init.uniform_(self.weight.bias, 0, 1)

    def orthogonalize(self, P):
        # P: [proj_dim, 3]
        Q, _ = torch.linalg.qr(P)
        return Q

    def forward(self, chiral_feats, q_mask=None):
        if self.chiral_encoder == "Linear":
            output_ = self.weight(chiral_feats)
            if not self.ecd:
                return output_, 0.
            else:
                return output_, -output_, 0.

        # chiral_feats: [B, Q, 9]
        x = chiral_feats[..., 0:3]   # [B, Q, 3]
        y = chiral_feats[..., 3:6]
        z = chiral_feats[..., 6:9]
        B, Q = x.shape[0], x.shape[1]
        P = self.proj_dim
        K = self.K

        W = self.weight  # [K, in_dim, proj_dim]
        if self.use_qr:
            # Transpose to [K, proj_dim, in_dim] for QR, then transpose back
            W_t = W.transpose(1, 2)  # [K, proj_dim, in_dim]
            Q_orth, _ = torch.linalg.qr(W_t)  # Q: [K, proj_dim, in_dim]
            W = Q_orth.transpose(1, 2)  # [K, in_dim, proj_dim]

        x = x.unsqueeze(2).expand(-1, -1, self.K, -1) 
        y = y.unsqueeze(2).expand(-1, -1, self.K, -1) 
        z = z.unsqueeze(2).expand(-1, -1, self.K, -1)
        W_exp = W.unsqueeze(0).unsqueeze(0).expand(B, Q, -1, -1, -1)  # [B, Q, K, in_dim, proj_dim]

        x_proj = torch.matmul(x.unsqueeze(3), W_exp).squeeze(3)  # [B, Q, K, proj_dim]
        y_proj = torch.matmul(y.unsqueeze(3), W_exp).squeeze(3)
        z_proj = torch.matmul(z.unsqueeze(3), W_exp).squeeze(3)

        # L2 normalize the projected vectors to avoid numerical instability
        x_proj = F.normalize(x_proj, p=2, dim=-1)
        y_proj = F.normalize(y_proj, p=2, dim=-1)
        z_proj = F.normalize(z_proj, p=2, dim=-1)

        V = torch.stack([x_proj, y_proj, z_proj], dim=-1)  # [B, Q, K, proj_dim, 3]
        # do not just padding for QR to compute
        if q_mask is None:
            BQK = B * Q * K
            V_batch = V.reshape(BQK, P, 3)
            V_batch = V_batch + 1e-8 * torch.randn_like(V_batch)  # Jitter for numerical stability
            Qmat, R = torch.linalg.qr(V_batch, mode='reduced')  # Qmat: [BQK, P, 3], R: [BQK, 3, 3]
            signed_vol_full = torch.linalg.det(R).view(B, Q, K)
        else:
            V_flat = V.reshape(B, Q, K, P, 3)  # [B, Q, K, P, 3]

            # mask flat
            mask_flat = q_mask.unsqueeze(-1).expand(-1, -1, K).reshape(-1)  # [B*Q*K]

            V_valid = V_flat.reshape(-1, P, 3)[mask_flat.bool()]  # [num_valid, P, 3]
            V_valid = V_valid + 1e-8 * torch.randn_like(V_valid)  # Jitter for numerically stable QR
            Qmat, R = torch.linalg.qr(V_valid, mode='reduced')  # [num_valid, P, 3], [num_valid, 3, 3]
            signed_vol_valid = torch.linalg.det(R)

            # Scatter back to full tensor
            signed_vol_full = torch.zeros(B*Q*K, device=V.device)
            signed_vol_full[mask_flat.bool()] = signed_vol_valid
            signed_vol_full = signed_vol_full.view(B, Q, K)  # [B, Q, K]


        # Orthogonalization regularization loss
        W_reg = W  # [K, in_dim, proj_dim]
        WWt = torch.matmul(W_reg, W_reg.transpose(-1, -2))  # [K, in_dim, in_dim]
        I = torch.eye(self.in_dim, device=W_reg.device).unsqueeze(0)  # [1, in_dim, in_dim]
        orth_reg = ((WWt - I)**2).sum()

        if not self.ecd:
            return signed_vol_full, orth_reg
        else:
            return signed_vol_full, -signed_vol_full, orth_reg

