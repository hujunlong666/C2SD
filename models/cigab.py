import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks

from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN


class GNNERM(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GNNERM, self).__init__()
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)

    def forward(self, batch, return_data="pred"):
        causal_pred, causal_rep = self.classifier(batch, get_rep=True)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")



class GNNPooling(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 pooling='asap',
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GNNPooling, self).__init__()
        if pooling.lower() == 'asap':
            # Cancel out the edge attribute when using ASAP pooling
            # since (1) ASAP not compatible with edge attr
            #       (2) performance of DrugOOD will not be affected w/o edge attr
            self.pool = ASAPooling(emb_dim, ratio, dropout=drop_ratio)
            edge_dim = -1
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.pooling = pooling

        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=emb_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)

    def forward(self, batched_data, return_data="pred"):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        device = x.device
        h = self.gnn_encoder(batched_data)
        edge_weight = None  #torch.ones(edge_index[0].size()).to(device)
        x, edge_index, causal_edge_weight, batch, perm = self.pool(h, edge_index, edge_weight=edge_weight, batch=batch)
        col, row = batched_data.edge_index
        node_mask = torch.zeros(batched_data.x.size(0)).to(device)
        node_mask[perm] = 1
        edge_mask = node_mask[col] * node_mask[row]
        if self.pooling.lower() == 'asap':
            # Cancel out the edge attribute when using ASAP pooling
            # since (1) ASAP not compatible with edge attr
            #       (2) performance of DrugOOD will not be affected w/o edge attr
            edge_attr = torch.ones(row.size()).to(device)

        # causal_x, causal_edge_index, causal_batch, _ = relabel(x, edge_index, batch)
        causal_x, causal_edge_index, causal_batch = x, edge_index, batch
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")


class CIGA(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 sigma_len=3):
        super(CIGA, self).__init__()
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)

        # self.gnn_encoder.reset_parameters()
        # self.classifier.reset_parameters()

        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)



    def split_graph(self,data, edge_score, ratio):
        # Adopt from GOOD benchmark to improve the efficiency
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):

            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]

            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)

            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)

            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()

            topk_perm = perm[mask]

            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = -edge_score[new_idx_drop]
        edge_score[new_idx_drop]*=-10
        new=edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight,edge_score)

    def forward(self, batch, return_data="pred", return_spu=False, debug=False,subgraph_avg=False,pred_edge_weight=None):

        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device

        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)

        edge_rep = torch.cat([h[row], h[col]], dim=-1)

        if subgraph_avg:
            pass
        else:
            pred_edge_weight = self.edge_att(edge_rep).view(-1)
        if self.ratio<0:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = (batch.edge_index, batch.edge_attr, pred_edge_weight), \
                (batch.edge_index, batch.edge_attr, pred_edge_weight)
        else:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight, modified_edge_weights) = self.split_graph(batch, pred_edge_weight, self.ratio)
        unmapped_casual_edge=causal_edge_index

        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

        # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)

        set_masks(causal_edge_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        if return_spu:
            spu_graph = DataBatch.Batch(batch=spu_batch,
                                         edge_index=spu_edge_index,
                                         x=spu_x,
                                         edge_attr=spu_edge_attr)
            set_masks(spu_edge_weight, self.classifier)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(spu_graph, get_rep=True)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
            clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)


        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
            if self.c_pool.lower() == "add":
                casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "max":
                casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "mean":
                casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
            else:
                raise Exception("Not implemented contrastive feature pooling")

            return causal_pred, casual_rep_from_feat
        elif return_data.lower() == "visual":
            return causal_graph,unmapped_casual_edge
        elif return_data.lower() == "viz":
            return pred_edge_weight
        elif return_data.lower() == "attention":
            return pred_edge_weight
        else:
            return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight

    # def get_dir_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
    #     (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
    #             (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
    #             pred_edge_weight = self.forward(batched_data,return_data="")
    #     criterion = nn.BCELoss()
    #     # if is_labeled == None:
    #         # is_labeled = torch.ones_like(labels, dtype=torch.bool, device=labels.device)
    #         # is_labeled = torch.ones(labels.size()).to(labels.device)
    #
    #     def get_comb_pred(predictor, causal_graph_x, spu_graph_x):
    #         # logits
    #         causal_logits = predictor.graph_pred_linear(causal_graph_x)  # (B, C)
    #         spu_logits = predictor.spu_mlp(spu_graph_x).detach()  # (B, 1) 或 (B, C)
    #
    #         # 二分类：取正类 logit
    #         if causal_logits.size(-1) == 2:
    #             causal_pos_logit = causal_logits[:, 1:2]
    #         else:  # (B,1)
    #             causal_pos_logit = causal_logits
    #
    #         # 概率相乘（两支都过 sigmoid）
    #         p = torch.sigmoid(causal_pos_logit) * torch.sigmoid(spu_logits)
    #         return p  # 概率 in (0,1)
    #     # def get_comb_pred(predictor, causal_graph_x, spu_graph_x):
    #     #     causal_pred = predictor.graph_pred_linear(causal_graph_x)
    #     #     spu_pred = predictor.spu_mlp(spu_graph_x).detach()
    #     #     return torch.sigmoid(spu_pred) * causal_pred
    #
    #     causal_graph = DataBatch.Batch(batch=causal_batch,
    #                                    edge_index=causal_edge_index,
    #                                    x=causal_x,
    #                                    edge_attr=causal_edge_attr)
    #     set_masks(causal_edge_weight, self.classifier)
    #     causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
    #     clear_masks(self.classifier)
    #
    #     spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
    #     set_masks(spu_edge_weight, self.classifier)
    #     spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
    #     clear_masks(self.classifier)
    #
    #     env_loss = torch.tensor([]).to(causal_rep.device)
    #     for spu in spu_rep:
    #         rep_out = get_comb_pred(self.classifier, causal_rep, spu)
    #         env_loss = torch.cat([env_loss, criterion(rep_out[is_labeled], labels[is_labeled]).unsqueeze(0)])
    #
    #     dir_loss = torch.var(env_loss * spu_rep.size(0)) + env_loss.mean()
    #
    #     if return_data.lower() == "pred":
    #         return get_comb_pred(causal_rep, spu_rep)
    #     elif return_data.lower() == "rep":
    #         return dir_loss, causal_pred, spu_pred, causal_rep
    #     else:
    #         return dir_loss, causal_pred

    def get_dir_loss(self, batched_data, labels, criterion=None, is_labeled=None, return_data="pred"):
        # 1) 前向拆分
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch), \
            pred_edge_weight = self.forward(batched_data, return_data="")

        # 2) 标签统一：(B,1) float；自动合法性掩码（仅保留 {0,1}）
        labels_f = labels.float().view(-1, 1)
        legal_mask = (labels_f == 0.0) | (labels_f == 1.0)  # (B,1) bool
        legal_mask = legal_mask.view(-1)  # (B,)

        # 3) 用户自带掩码（可选），与合法掩码求交
        if is_labeled is None:
            mask = legal_mask
        else:
            mask = legal_mask & is_labeled.view(-1).to(torch.bool)

        # 若没有可监督的样本，直接返回 0（避免后面 stack 空列表等错误）
        if mask.sum() == 0:
            zero = torch.zeros((), device=labels_f.device)
            if return_data.lower() == "rep":
                # 需要返回4元组时，补上占位
                return zero, zero, zero, zero
            elif return_data.lower() == "pred":
                return torch.zeros_like(labels_f)  # (B,1)
            else:
                return zero, zero

        # 4) Loss：默认 BCELoss（输入是概率）
        if criterion is None:
            criterion = nn.BCELoss()

        # 5) 工具：把任意 logits 规范成 (B,1) 的“正类 logit”
        def _pos_logit_2d(t: torch.Tensor) -> torch.Tensor:
            # 支持 (B,), (B,1), (B,2), (B,C)
            if t.dim() == 1:
                t = t.unsqueeze(1)  # (B,) -> (B,1)
            if t.size(1) >= 2:
                t = t[:, 1:2]  # 取正类列
            return t  # (B,1)

        # 6) 组合概率（稳定 & clamp）
        def get_comb_prob(predictor, causal_graph_x, spu_graph_x):
            causal_logits = predictor.graph_pred_linear(causal_graph_x)
            spu_logits = predictor.spu_mlp(spu_graph_x).detach()  # 只让梯度走因果分支

            causal_pos_logit = _pos_logit_2d(causal_logits)  # (B,1)
            spu_pos_logit = _pos_logit_2d(spu_logits)  # (B,1)

            p = torch.sigmoid(causal_pos_logit) * torch.sigmoid(spu_pos_logit)  # (B,1)

            # clamp 防止极值/NaN（BCELoss 需要 [0,1] 输入）
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            # 若仍出现数值问题，直接替换为 0.5，避免 CUDA assert
            p = torch.where(torch.isfinite(p), p, torch.full_like(p, 0.5))
            return p

        # 7) 拿到因果/虚假表示
        causal_graph = DataBatch.Batch(batch=causal_batch, edge_index=causal_edge_index,
                                       x=causal_x, edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index,
                                    x=spu_x, edge_attr=spu_edge_attr)
        set_masks(spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)

        # spu_rep 可能是张量或序列，统一为 list
        spu_rep_list = [spu_rep] if isinstance(spu_rep, torch.Tensor) else list(spu_rep)

        # 8) 逐“环境”损失：方差 + 均值
        env_losses = []
        for spu in spu_rep_list:
            p = get_comb_prob(self.classifier, causal_rep, spu)  # (B,1)
            loss_i = criterion(p[mask], labels_f[mask])  # BCELoss 输入∈[0,1]
            env_losses.append(loss_i)

        env_loss = torch.stack(env_losses)  # (num_env,)
        dir_loss = torch.var(env_loss * len(env_losses)) + env_loss.mean()

        # 9) 返回
        if return_data.lower() == "pred":
            with torch.no_grad():
                ps = [get_comb_prob(self.classifier, causal_rep, spu) for spu in spu_rep_list]  # list of (B,1)
                p_avg = torch.mean(torch.stack(ps, dim=0), dim=0)  # (B,1)
            return p_avg
        elif return_data.lower() == "rep":
            return dir_loss, causal_pred, spu_pred, causal_rep
        else:
            return dir_loss, causal_pred

    # def get_dir_loss(self, batched_data, labels, criterion=None, is_labeled=None, return_data="pred"):
    #     """
    #     返回：
    #       - return_data="rep"  -> (dir_loss, causal_pred, spu_pred, causal_rep)
    #       - return_data="pred" -> (B,1) 的组合概率（各“环境”平均）
    #       - 其它                -> (dir_loss, causal_pred)
    #     说明：
    #       - 使用 BCELoss（输入是概率），组合为 p = σ(causal_pos_logit) * σ(spu_pos_logit)
    #       - 自动过滤非法标签（非{0,1}），并对概率 clamp 防止数值问题
    #     """
    #     # 1) 前向拆分
    #     (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
    #         (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch), \
    #         pred_edge_weight = self.forward(batched_data, return_data="")
    #
    #     # 2) 标签统一：(B,1) float；仅保留 {0,1}
    #     labels_f = labels.float().view(-1, 1)
    #     legal_mask = ((labels_f == 0.0) | (labels_f == 1.0)).view(-1)  # (B,)
    #
    #     # 3) 用户掩码（可选）
    #     if is_labeled is None:
    #         mask = legal_mask
    #     else:
    #         mask = legal_mask & is_labeled.view(-1).to(torch.bool)
    #
    #     # 空掩码兜底
    #     if mask.sum() == 0:
    #         zero = torch.zeros((), device=labels_f.device)
    #         if return_data.lower() == "rep":
    #             return zero, zero, zero, zero
    #         elif return_data.lower() == "pred":
    #             return torch.zeros_like(labels_f)  # (B,1)
    #         else:
    #             return zero, zero
    #
    #     # 4) Loss：默认 BCELoss（输入应是概率）
    #     if criterion is None:
    #         criterion = nn.BCELoss()
    #
    #     # 5) 工具：统一为 (B,1) 的“正类 logit”
    #     def _pos_logit_2d(t: torch.Tensor) -> torch.Tensor:
    #         if t.dim() == 1:
    #             t = t.unsqueeze(1)  # (B,) -> (B,1)
    #         if t.size(1) >= 2:
    #             t = t[:, 1:2]  # 取正类
    #         return t  # (B,1)
    #
    #     # 6) 组合概率：σ(causal)*σ(spurious)（含 clamp 与 NaN 兜底）
    #     def get_comb_prob(predictor, causal_graph_x, spu_graph_x):
    #         causal_logits = predictor.graph_pred_linear(causal_graph_x)
    #         spu_logits = predictor.spu_mlp(spu_graph_x).detach()  # 只让梯度走因果分支
    #
    #         causal_pos_logit = _pos_logit_2d(causal_logits)  # (B,1)
    #         spu_pos_logit = _pos_logit_2d(spu_logits)  # (B,1)
    #
    #         p = torch.sigmoid(causal_pos_logit) * torch.sigmoid(spu_pos_logit)  # (B,1)
    #         p = torch.clamp(p, 1e-6, 1 - 1e-6)
    #         p = torch.where(torch.isfinite(p), p, torch.full_like(p, 0.5))
    #         return p
    #
    #     # 7) 拿到因果/虚假表示
    #     causal_graph = DataBatch.Batch(batch=causal_batch, edge_index=causal_edge_index,
    #                                    x=causal_x, edge_attr=causal_edge_attr)
    #     set_masks(causal_edge_weight, self.classifier)
    #     causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
    #     clear_masks(self.classifier)
    #
    #     spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index,
    #                                 x=spu_x, edge_attr=spu_edge_attr)
    #     set_masks(spu_edge_weight, self.classifier)
    #     spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
    #     clear_masks(self.classifier)
    #
    #     # 统一 spu_rep 为 list
    #     spu_rep_list = [spu_rep] if isinstance(spu_rep, torch.Tensor) else list(spu_rep)
    #
    #     # 8) 逐“环境”损失：方差 + 均值（无偏置放大）
    #     env_losses = []
    #     for spu in spu_rep_list:
    #         p = get_comb_prob(self.classifier, causal_rep, spu)  # (B,1)
    #         loss_i = criterion(p[mask], labels_f[mask])  # BCELoss 概率
    #         env_losses.append(loss_i)
    #
    #     env_loss = torch.stack(env_losses)  # (num_env,)
    #     dir_loss = torch.var(env_loss, unbiased=False) + env_loss.mean()
    #
    #     # 9) 返回
    #     if return_data.lower() == "pred":
    #         with torch.no_grad():
    #             ps = [get_comb_prob(self.classifier, causal_rep, spu) for spu in spu_rep_list]  # list of (B,1)
    #             p_avg = torch.mean(torch.stack(ps, dim=0), dim=0)  # (B,1)
    #         return p_avg
    #     elif return_data.lower() == "rep":
    #         return dir_loss, causal_pred, spu_pred, causal_rep
    #     else:
    #         return dir_loss, causal_pred