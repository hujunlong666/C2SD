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
from torch_scatter import scatter_max


class BiasedInfomaxModule(nn.Module):
    def __init__(self, node_dim, graph_dim):
        super(BiasedInfomaxModule, self).__init__()
        # 互信息判别器：使用双线性层评估节点局部表示与图全局表示的相似度
        self.discriminator = nn.Bilinear(node_dim, graph_dim, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, node_embeddings, Z_s, node_causal_weights, batch_index):
        """
        node_embeddings: [N, node_dim]
        Z_s: [Batch, graph_dim]
        node_causal_weights: [N]
        batch_index: [N]
        """
        Z_s_expanded = Z_s[batch_index] # [N, graph_dim]
        scores = self.discriminator(node_embeddings, Z_s_expanded) # [N, 1]
        
        pseudo_labels = torch.ones_like(scores)
        node_causal_weights = node_causal_weights.view(-1, 1)
        
        loss_spurious = self.bce_loss(scores, pseudo_labels) * (1 - node_causal_weights)
        loss_causal = self.bce_loss(scores, torch.zeros_like(scores)) * node_causal_weights
        
        loss_biased_infomax = torch.mean(loss_spurious + loss_causal)
        return loss_biased_infomax



class SupConLoss(nn.Module):
    """监督对比学习损失 (带防 NaN 保护机制)"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 增加微小数值并安全截断，防止特征为全 0 导致后续计算崩溃
        features = features + 1e-8
        features = F.normalize(features, p=2, dim=1, eps=1e-12)

        # 构建相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 为了增强数值稳定性，减去每行的最大值 (不影响 softmax 的梯度)
        # 这可以防止 log_softmax 时出现极端的 logits 导致 NaN
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        # 用 -1e9 取代 -inf，避免某些 PyTorch 版本在全负极大值时溢出产生 NaN
        sim_matrix = sim_matrix.masked_fill(mask_self, -1e9)

        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T) & ~mask_self

        has_pos = mask_pos.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_prob = F.log_softmax(sim_matrix, dim=1)

        loss_per_sample = -(log_prob * mask_pos.float()).sum(dim=1) / mask_pos.float().sum(dim=1).clamp(min=1.0)
        loss = loss_per_sample[has_pos].mean()

        # 终极保护层: 若 loss 为 NaN (极少见的特征坍缩情况)，则使其梯变为0
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

class GNNPooling(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=3,
                 ratio=0.25,
                 pooling='mean',
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
                 num_layers=3,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=False,
                 residual=True,
                 drop_ratio=0.3,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="conv",
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

        self.cf_k = 3  # 默认生成 k 组
        self.cf_detach_zs = True  # 建议 True：只让正则通过 Zc 反传
        self.cf_eps = 1e-6
        self.cf_head = nn.Sequential(
            nn.Linear(2 * self.c_dim, 2 * self.c_dim),
            nn.ReLU(),
            nn.Linear(2 * self.c_dim, out_dim)
        )

        # Biased Infomax Module
        self.biased_infomax = BiasedInfomaxModule(node_dim=emb_dim, graph_dim=self.c_dim)

        # Supervised Contrastive Loss on Z_c
        self.supcon_loss = SupConLoss(temperature=0.07)

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

    def _sample_gumbel(self, shape, device, eps=1e-20):
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)

    def _gumbel_sigmoid(self, logits, tau=1.0, hard=False, training=True, eps=1e-20):
        """
        logits: R^E, 未经过 sigmoid 的打分（如果你的 edge_att 输出已经是(0,1)，下面 forward 里我会自动转 logits）
        """
        if training:
            g = self._sample_gumbel(logits.shape, logits.device, eps=eps)
            y = torch.sigmoid((logits + g) / tau)
        else:
            y = torch.sigmoid(logits)

        if not hard:
            return y

        # straight-through hard gate（可选）
        y_hard = (y > 0.5).float()
        return y_hard - y.detach() + y

    # def forward(self, batch, return_data="pred", return_spu=False, debug=False,subgraph_avg=False,pred_edge_weight=None):
    def forward(self, batch, return_data="pred", debug=False, subgraph_avg=False,
                    pred_edge_weight=None):
        # obtain the graph embeddings from the featurizer GNN encoder
        # #{#############################################
        # h = self.gnn_encoder(batch)
        # device = h.device
        #
        # # using edge-level attetion
        # row, col = batch.edge_index
        # if batch.edge_attr == None:
        #     batch.edge_attr = torch.ones(row.size(0)).to(device)
        #
        # edge_rep = torch.cat([h[row], h[col]], dim=-1)
        #
        # if subgraph_avg:
        #     pass
        # else:
        #     pred_edge_weight = self.edge_att(edge_rep).view(-1)
        # if self.ratio<0:
        #     (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #         (spu_edge_index, spu_edge_attr, spu_edge_weight) = (batch.edge_index, batch.edge_attr, pred_edge_weight), \
        #         (batch.edge_index, batch.edge_attr, pred_edge_weight)
        # else:
        #     (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #         (spu_edge_index, spu_edge_attr, spu_edge_weight, modified_edge_weights) = self.split_graph(batch, pred_edge_weight, self.ratio)
        # unmapped_casual_edge=causal_edge_index
        #
        # if self.c_in.lower() == "raw":
        #     causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
        #     spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        # else:
        #     causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
        #     spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)
        #
        # # obtain \hat{G_c}
        # causal_graph = DataBatch.Batch(batch=causal_batch,
        #                                edge_index=causal_edge_index,
        #                                x=causal_x,
        #                                edge_attr=causal_edge_attr)
        #
        # set_masks(causal_edge_weight, self.classifier)
        # # obtain predictions with the classifier based on \hat{G_c}
        # # causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        # causal_rep = self.classifier.forward_rep(causal_graph)
        # clear_masks(self.classifier)
        #
        #
        # spu_graph = DataBatch.Batch(batch=spu_batch,
        #                                 edge_index=spu_edge_index,
        #                                 x=spu_x,
        #                                 edge_attr=spu_edge_attr)
        # #}##################################################################################
        h = self.gnn_encoder(batch)
        device = h.device

        row, col = batch.edge_index
        if batch.edge_attr is None:
            batch.edge_attr = torch.ones(row.size(0), device=device)
        else:
            batch.edge_attr = batch.edge_attr.to(device)

        edge_rep = torch.cat([h[row], h[col]], dim=-1)

        if not subgraph_avg:
            pred_edge_weight = self.edge_att(edge_rep).view(-1)
        # else: 原来 pass，这里不动（保持你可能从外部传 pred_edge_weight 的用法）

        # ===== 关键：Gumbel-Sigmoid 软门控（可微）=====
        tau = getattr(self, "tau", 1.0)  #
        hard_gate = getattr(self, "hard_gate", True)

        # 直接使用 edge_att 的输出作为 logits，避免极小概率的误判（当预测值恰好在 [0, 1] 之间时导致的梯度爆炸）
        edge_logits = pred_edge_weight

        gate = self._gumbel_sigmoid(
            edge_logits, tau=tau, hard=hard_gate, training=self.training
        )  # shape [E], in (0,1) or ST-hard

        # print("hard_gate", getattr(self, "hard_gate", True))
        # print("gate unique approx:", (gate.detach().round().unique()[:10]))
        # print("gate min/mean/max:", gate.min().item(), gate.mean().item(), gate.max().item())

        causal_edge_weight = gate
        spu_edge_weight = 1.0 - gate

        # （可选但强烈建议）加一个“预算正则”，让每张图平均 gate 接近 ratio，避免 gate 全 1 或全 0
        # 不改变 forward 入参，只是在 extra 里返回一个 loss_budget，你决定是否加到总损失
        loss_budget = torch.tensor(0.0, device=device)
        if getattr(self, "disable_budget", False):
            pass  # ablation: skip budget regularization
        elif getattr(self, "ratio", -1) is not None and self.ratio >= 0:
            edge_graph = batch.batch[row]  # 每条边属于哪个图（按 source node 的 batch）
            num_graphs = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 1
            sum_gate = torch.zeros(num_graphs, device=device).scatter_add_(0, edge_graph, gate)
            cnt = torch.zeros(num_graphs, device=device).scatter_add_(0, edge_graph, torch.ones_like(gate))
            mean_gate = sum_gate / (cnt + 1e-12)
            loss_budget = ((mean_gate - float(self.ratio)) ** 2).mean()
        ################软算损失硬mask
        # # 1) 永远先算 soft gate
        # gate_soft = self._gumbel_sigmoid(edge_logits, tau=tau, hard=False, training=self.training)
        #
        # # 2) 如果要二值前向，用 ST 得到 gate（用于 mask）
        # if hard_gate:
        #     gate_hard = (gate_soft > 0.5).float()
        #     gate = gate_hard - gate_soft.detach() + gate_soft
        # else:
        #     gate = gate_soft
        #
        # causal_edge_weight = gate
        # spu_edge_weight = 1.0 - gate
        #
        # # 3) budget 一律用 gate_soft（更稳）
        # edge_graph = batch.batch[row]
        # num_graphs = int(edge_graph.max().item()) + 1 if edge_graph.numel() > 0 else 1
        # sum_gate = torch.zeros(num_graphs, device=device).scatter_add_(0, edge_graph, gate_soft)
        # cnt = torch.zeros(num_graphs, device=device).scatter_add_(0, edge_graph, torch.ones_like(gate_soft))
        # mean_gate = sum_gate / (cnt + 1e-12)
        # loss_budget = ((mean_gate - float(self.ratio)) ** 2).mean()

        # ===== 不再 split_graph / 不再 relabel：两条分支都在原图上跑，但用不同 mask =====
        if self.c_in.lower() == "raw":
            causal_x = batch.x
            spu_x = batch.x
        else:
            causal_x = h
            spu_x = h

        causal_graph = DataBatch.Batch(
            batch=batch.batch,
            edge_index=batch.edge_index,
            x=causal_x,
            edge_attr=batch.edge_attr
        )

        spu_graph = DataBatch.Batch(
            batch=batch.batch,
            edge_index=batch.edge_index,
            x=spu_x,
            edge_attr=batch.edge_attr
        )
        set_masks(causal_edge_weight, self.classifier)
        causal_rep = self.classifier.forward_rep(causal_graph)
        clear_masks(self.classifier)
        ############################################################################################


        set_masks(spu_edge_weight, self.classifier)
        if self.s_rep.lower() == "conv":
            spu_pred, spu_rep = self.classifier.get_spu_pred_forward(spu_graph, get_rep=True)
            # print("convconv")
        else:
            spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
            # print("nonconv")

        clear_masks(self.classifier)
        # causal_pred = (causal_pred, spu_pred)

        causal_pred = self.classifier.forward_cls(causal_rep)

        y = batch.y.view(-1).long()
        # print("yyyyyyyy", y.shape)
        Z_c = causal_rep  # 因果图的表征
        Z_n = spu_rep  # 虚假图的表征

        # 计算节点的因果概率权重 (使用与其相连边的最大权重)
        node_cw_row, _ = scatter_max(gate, row, dim=0, dim_size=batch.x.size(0))
        node_cw_col, _ = scatter_max(gate, col, dim=0, dim_size=batch.x.size(0))
        node_causal_weights = torch.max(node_cw_row, node_cw_col)
        
        # 计算 Biased Infomax loss
        loss_bi = self.biased_infomax(h, Z_n, node_causal_weights, batch.batch)

        # ===== 核心：反事实头必须 detach Z_c =====
        Zc_detach = Z_c.detach()

        mu_Zn = Z_n.mean(dim=-1, keepdim=True)
        std_Zn = Z_n.std(dim=-1, keepdim=True) + 1e-6

        cf_logits_list = []

        for _ in range(self.cf_k):
            idx = torch.randperm(Z_n.size(0))
            Z_pern = Z_n[idx]

            mu_p = Z_pern.mean(dim=-1, keepdim=True)
            std_p = Z_pern.std(dim=-1, keepdim=True) + 1e-6

            Z_smv = std_p * (Z_n - mu_Zn) / std_Zn + mu_p



            Z_pern_combined = torch.cat([Zc_detach, Z_pern], dim=-1)
            logits_pern = self.cf_head(Z_pern_combined)  # [B, C]
            cf_logits_list.append(logits_pern)

            # Style-Mixing (can be disabled for ablation)
            if not getattr(self, "disable_stylemix", False):
                Z_smv_combined = torch.cat([Zc_detach, Z_smv], dim=-1)
                logits_smv = self.cf_head(Z_smv_combined)  # [B, C]
                cf_logits_list.append(logits_smv)

        probs = [F.softmax(l, dim=-1) for l in cf_logits_list]  # list of [B,C]

        p_mean = torch.stack(probs, dim=0).mean(dim=0).detach()  # [B,C]

        loss_cons = 0.0
        for l in cf_logits_list:
            loss_cons += F.kl_div(
                F.log_softmax(l, dim=-1),
                p_mean,
                reduction="batchmean"
            )


        anchor_logits = self.cf_head(
            torch.cat([Zc_detach, Z_n.detach()], dim=-1)
        )
        loss_anchor = F.cross_entropy(anchor_logits, y)

        loss_cons = loss_cons / len(cf_logits_list)

        # ===== 监督对比学习损失（作用在因果表征 Z_c 上）=====
        loss_supcon = self.supcon_loss(Z_c, y)

        extra = {
            "loss_cons": loss_cons,
            "loss_anchor": loss_anchor,
            "loss_budget": loss_budget,
            "loss_bi": loss_bi,
            "loss_supcon": loss_supcon,
        }



        # #{
        # Z_cat = torch.cat([causal_rep, spu_rep.detach()], dim=-1)
        # causal_pred = self.cf_head(Z_cat)
        #
        # y = batch.y.view(-1).long()
        # # print("yyyyyyyy", y.shape)
        # Z_c = causal_rep  # 因果图的表征
        # Z_n = spu_rep  # 虚假图的表征
        #
        #
        # mu_Zn = Z_n.mean(dim=0, keepdim=True)
        # std_Zn = Z_n.std(dim=0, keepdim=True) + 1e-6
        #
        #
        # cf_logits_list = []
        # cf_risk_list = []
        #
        # for _ in range(self.cf_k):
        #     idx = torch.randperm(Z_n.size(0))
        #     Z_pern = Z_n[idx]
        #
        #
        #     mu_p = Z_pern.mean(dim=0, keepdim=True)
        #     std_p = Z_pern.std(dim=0, keepdim=True) + 1e-6
        #
        #     Z_smv = std_p * (Z_n - mu_Zn) / std_Zn + mu_p
        #
        #     if self.cf_detach_zs:
        #         Z_pern = Z_pern.detach()
        #         Z_smv = Z_smv.detach()
        #
        #     Z_pern_combined = torch.cat([Z_c, Z_pern], dim=-1)
        #     Z_smv_combined = torch.cat([Z_c, Z_smv], dim=-1)
        #
        #     logits_pern = self.cf_head(Z_pern_combined)  # [B, C]
        #     logits_smv = self.cf_head(Z_smv_combined)  # [B, C]
        #
        #     cf_logits_list.extend([logits_pern, logits_smv])
        #
        #
        # for logits in cf_logits_list:
        #     cf_risk_list.append(F.cross_entropy(logits, y))
        #
        # risks = torch.stack(cf_risk_list)  # [2k]
        # loss_cf = risks.mean()
        # loss_vrex = torch.var(risks, unbiased=False)
        #
        #
        # # loss_cons: prediction consistency across environments
        # probs = [F.softmax(l, dim=-1) for l in cf_logits_list]  # list of [B,C]
        # p_mean = torch.stack(probs, dim=0).mean(dim=0).detach()  # [B,C]
        #
        # loss_cons = 0.0
        # for l in cf_logits_list:
        #     loss_cons = loss_cons + F.kl_div(F.log_softmax(l, dim=-1), p_mean, reduction="batchmean")
        # loss_cons = loss_cons / len(cf_logits_list)
        # # print("Causal representation dimension:", causal_rep.shape)
        # #
        # # # 打印虚假表征的维度
        # # print("Spurious representation dimension:", spu_rep.shape)
        #
        # extra = {
        #     "loss_cf": loss_cf,
        #     "loss_vrex": loss_vrex,
        #     "loss_cons": loss_cons,
        #     # 可选：调试用
        #     # "cf_logits_list": cf_logits_list,
        # }
        #}


        if return_data.lower() == "pred":
            return causal_pred, extra
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        # elif return_data.lower() == "feat":
        #     causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
        #     if self.c_pool.lower() == "add":
        #         casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
        #     elif self.c_pool.lower() == "max":
        #         casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
        #     elif self.c_pool.lower() == "mean":
        #         casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
        #     else:
        #         raise Exception("Not implemented contrastive feature pooling")
        #
        #     return causal_pred, casual_rep_from_feat
        # elif return_data.lower() == "visual":
        #     return causal_graph,unmapped_casual_edge
        # elif return_data.lower() == "viz":
        #     return pred_edge_weight
        # elif return_data.lower() == "attention":
        #     return pred_edge_weight
        # else:
        #     return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
        #         (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
        #         pred_edge_weight



