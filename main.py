import os
import random
from datetime import datetime
import json
import torch


_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import argparse
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader as GeoDataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm
import torch.nn.functional as F
from models.ciga import CIGA
from torch_geometric.data import Data
import numpy as np
from utils.util import load_pt_as_dataset, _FromMemory
from glob import glob
def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN model on OGB HIV dataset")
    parser.add_argument('--runs', type=int, default=1, help="在一次脚本运行中重复训练的次数")
    # General hyperparameters
    parser.add_argument('--dataset', type=str, default='ogbg-molbace', help="Dataset name")
    # ogbg-molbace ogbg-molbbbp ogbg-molhiv drugood
    # 当 --dataset=drugood 时生效：
    parser.add_argument('--drugood_dir', type=str, default='dataset/DrugOOD',
                        help="DrugOOD 数据所在目录")
    parser.add_argument('--drugood_prefix', type=str, default='drugood_lbap_core_ec50',
                        help="文件名前缀（不含 OOD 和 split），例如 drugood_lbap_core_ec50")
    parser.add_argument('--drugood_ood', type=str, default='size',
                        help="OOD 类型，例如 assay/ scaffold/ size 等")

    # 可选：单独覆写具体文件名（给不规则命名或特殊实验预留）
    parser.add_argument('--drugood_train_pt', type=str, default='',
                        help="显式指定 train .pt 文件名（可相对 drugood_dir 或绝对路径）")
    parser.add_argument('--drugood_val_pt', type=str, default='',
                        help="显式指定 val .pt 文件名（可相对 drugood_dir 或绝对路径）")
    parser.add_argument('--drugood_test_pt', type=str, default='',
                        help="显式指定 test .pt 文件名（可相对 drugood_dir 或绝对路径）")





    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--gnn_type', type=str, default='gin', help="Type of GNN (e.g., 'gin', 'graphsage')")
    parser.add_argument('--emb_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the GNN")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help="Device to train on")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility, -1 for random")
    # ecassay:2759777559 0.8
    parser.add_argument('--drop_ratio', type=float, default=0.3, help="Dropout ratio")
    parser.add_argument('--use_mask', type=int, default=1, help="Whether to use edge masks")
    parser.add_argument('--graph_pooling', type=str, default='mean', choices=['mean', 'max', 'sum'],
                        help="Graph pooling method (mean/max/sum)")

    # Model-specific parameters
    parser.add_argument('--virtual_node', type=int, default=1, choices=[0, 1], help="Whether to use virtual node")
    parser.add_argument('--residual', type=int, default=1, choices=[0, 1], help="Whether to se residual connections")
    parser.add_argument('--JK', type=str, default='last', choices=['last', 'all'], help="Jumping Knowledge (JK) layer")

    # CIGA-specific parameters
    parser.add_argument('--ratio', type=float, default=0.4, help="Ratio for causal subgraph extraction")
    parser.add_argument('--c_dim', type=int, default=-1, help="Classifier dimension (-1 for same as emb_dim)")
    parser.add_argument('--c_in', type=str, default='raw', choices=['raw', 'processed'], help="Classifier input type")
    parser.add_argument('--c_rep', type=str, default='rep', choices=['rep', 'feat'],
                        help="Classifier representation type")
    parser.add_argument('--c_pool', type=str, default='mean', choices=['add', 'mean', 'max'],
                        help="Contrastive pooling method")
    parser.add_argument('--s_rep', type=str, default='conv', choices=['rep', 'conv'], help="SPU representation type")
    parser.add_argument('--sigma_len', type=int, default=3, help="Length of sigma for dir loss")

    # Training control
    parser.add_argument('--eval_interval', type=int, default=10, help="Evaluation interval (epochs)")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help="Val AUC 连续多少轮未提高时早停，0 表示禁用")

    # Ablation & loss weights
    parser.add_argument('--alpha', type=float, default=0.1, help="Weight for alpha (unused currently)")
    parser.add_argument('--beta', type=float, default=0.1, help="Weight for anchor loss")
    parser.add_argument('--gamma', type=float, default=0.5, help="Weight for consistency loss")
    parser.add_argument('--delta', type=float, default=5.0, help="Weight for budget loss")
    parser.add_argument('--disable_budget', action='store_true', help="Disable budget regularization (ablation)")
    parser.add_argument('--disable_stylemix', action='store_true', help="Disable style-mixing, only use permutation (ablation)")
    parser.add_argument('--cf_k', type=int, default=3, help="Number of counterfactual groups")

    args = parser.parse_args()
    if args.early_stop_patience < 0:
        parser.error('--early_stop_patience 必须大于等于 0')
    return args


def _resolve_path(root_dir: str, maybe_rel_or_abs: str) -> str:
    """支持绝对/相对路径；若传空字符串则返回空。"""
    if not maybe_rel_or_abs:
        return ''
    if os.path.isabs(maybe_rel_or_abs):
        return maybe_rel_or_abs
    return os.path.join(root_dir, maybe_rel_or_abs)

def build_drugood_paths(root_dir: str,
                        prefix: str,
                        ood: str,
                        train_override: str = '',
                        val_override: str = '',
                        test_override: str = ''):
    """
    生成 train/val/test 的 .pt 路径，自动兼容如下两种命名：
    1) {prefix}_{ood}_{split}.pt       -> 常见于 val/test
    2) {prefix}_{split}.pt             -> 常见于 train（train 的后缀为 assay_train.pt）
    同时支持通过 *_override 显式指定文件名。
    """
    def join(p):  # 相对 -> 绝对
        return p if os.path.isabs(p) else os.path.join(root_dir, p)

    # 如果提供了 override，直接使用（支持相对/绝对路径）
    if train_override:
        train_candidates = [join(train_override)]
    else:
        # 修改：train 的文件名使用 'assay_train.pt' 而非 'assay_ood_train.pt'
        train_candidates = [
            os.path.join(root_dir, f"{prefix}_{ood}_train.pt"),  # 带 ood
            os.path.join(root_dir, f"{prefix}_train.pt"),        # 不带 ood（你的情况）
        ]

    if val_override:
        val_candidates = [join(val_override)]
    else:
        val_candidates = [
            os.path.join(root_dir, f"{prefix}_{ood}_ood_val.pt"),
            os.path.join(root_dir, f"{prefix}_val.pt"),
        ]

    if test_override:
        test_candidates = [join(test_override)]
    else:
        test_candidates = [
            os.path.join(root_dir, f"{prefix}_{ood}_ood_test.pt"),
            os.path.join(root_dir, f"{prefix}_test.pt"),
        ]

    def pick_existing(cands, split):
        for p in cands:
            if os.path.exists(p):
                return p
        # 兜底：模糊匹配（大小写不敏感）
        stem_options = [os.path.splitext(os.path.basename(c))[0].lower() for c in cands]
        pts = glob(os.path.join(root_dir, "*.pt"))
        for c in pts:
            base = os.path.splitext(os.path.basename(c))[0].lower()
            if any(stem in base for stem in stem_options):
                return c
        # 明确报错，便于排查
        raise FileNotFoundError(
            f"未找到 {split} 文件，尝试过：\n  " + "\n  ".join(cands)
        )

    train_pt = pick_existing(train_candidates, 'train')
    val_pt   = pick_existing(val_candidates,   'val')
    test_pt  = pick_existing(test_candidates,  'test')

    return train_pt, val_pt, test_pt
class ToFloatTransform(object):
    def __call__(self, data: Data):
        data.x = data.x.float()
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        return data

def train_one_epoch(model, loader, optimizer, device,
                    alpha=0.1, beta=0.1, gamma=0.5, delta=1.0):
    model.train()
    total_loss, total_graphs = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        y_long = batch.y.view(-1).long()

        # 现在 forward 返回 (causal_logits, extra)
        causal_logits, extra = model(batch, return_data="pred")

        ce_loss = F.cross_entropy(causal_logits, y_long)

        loss_budget = extra["loss_budget"]
        loss_cons = extra["loss_cons"]
        loss_anchor = extra["loss_anchor"]
        loss = ce_loss + gamma * loss_cons + beta * loss_anchor + delta * loss_budget

        if torch.isnan(loss):
            print(f"[DEBUG NaN] ce_loss={ce_loss.item():.4f}, loss_cons={loss_cons.item() if isinstance(loss_cons, torch.Tensor) else loss_cons:.4f}, "
                  f"loss_anchor={loss_anchor.item():.4f}, loss_budget={loss_budget.item():.4f}, bsz={bsz}")


        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bsz = y_long.size(0)
        total_loss += loss.item() * bsz
        total_graphs += bsz

    return total_loss / max(total_graphs, 1)




@torch.no_grad()
def eval_metrics(model, loader, device, evaluator):
    model.eval()
    y_true_list, y_prob_list, y_pred_list = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch, return_data="pred")
        logits = out[0] if isinstance(out, (tuple, list)) else out  #
        # 数值兜底：消灭 NaN/Inf，再稳定 softmax
        if not torch.isfinite(logits).all():
            print("[WARN] Non-finite logits detected during eval; sanitizing with nan_to_num.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        prob = torch.softmax(logits, dim=-1)
        prob = torch.clamp(prob, 1e-7, 1 - 1e-7)

        prob_pos = prob[:, 1:2].detach().cpu()  # (B,1)
        pred_class = prob.argmax(dim=-1, keepdim=True).cpu()
        y_true = batch.y.view(-1, 1).detach().cpu()

        prob_pos = torch.nan_to_num(prob_pos, nan=0.5, posinf=1.0, neginf=0.0)

        y_true_list.append(y_true)
        y_prob_list.append(prob_pos)
        y_pred_list.append(pred_class)

    y_true = torch.cat(y_true_list, dim=0)
    y_prob = torch.cat(y_prob_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    y_prob = torch.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    result = evaluator.eval({"y_true": y_true, "y_pred": y_prob})
    rocauc = result["rocauc"]
    acc = (y_pred == y_true).float().mean().item()
    return rocauc, acc

class SimpleBinaryEvaluator:
    def __init__(self):
        try:
            from sklearn.metrics import roc_auc_score  # noqa: F401
            self._ok = True
        except Exception:
            self._ok = False

    def eval(self, input_dict):
        y_true = input_dict["y_true"].view(-1).numpy()
        y_pred = input_dict["y_pred"].view(-1).numpy()
        out = {"rocauc": float('nan')}
        if self._ok:
            from sklearn.metrics import roc_auc_score
            try:
                out["rocauc"] = float(roc_auc_score(y_true, y_pred))
            except Exception:
                pass
        return out



def main():
    args = parse_args()

    # ====== 固定数据集，只加载一次 ======
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dataset == 'drugood':
        save_dir = os.path.join(args.save_dir, f"{args.drugood_prefix}_{args.drugood_ood}_{args.dataset}_{timestamp}")
    else:
        save_dir = os.path.join(args.save_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    if args.dataset in ['ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molhiv']:
        dataset = PygGraphPropPredDataset(name=args.dataset, transform=ToFloatTransform())

        print(f"加载的数据集名称：{args.dataset}")
        print(f"数据集的节点特征维度：{dataset.num_node_features}")
        print(f"数据集的类别数：{dataset.num_classes}")

        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx['train']]
        val_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]

        print(f"训练集大小：{len(train_dataset)}")
        print(f"验证集大小：{len(val_dataset)}")
        print(f"测试集大小：{len(test_dataset)}")

        input_dim = dataset.num_node_features
        output_dim = dataset.num_classes
        evaluator = Evaluator(name=args.dataset)

    elif args.dataset.lower() == 'drugood':
        root_dir = args.drugood_dir
        train_pt, val_pt, test_pt = build_drugood_paths(
            root_dir=root_dir,
            prefix=args.drugood_prefix,  # 例如 drugood_lbap_core_ec50
            ood=args.drugood_ood,  # 例如 assay_ood / scaffold_ood
            train_override=args.drugood_train_pt,
            val_override=args.drugood_val_pt,
            test_override=args.drugood_test_pt
        )
        print(f"[DrugOOD] train: {train_pt}")
        print(f"[DrugOOD] val  : {val_pt}")
        print(f"[DrugOOD] test : {test_pt}")

        transform = ToFloatTransform()
        train_dataset = load_pt_as_dataset(train_pt, transform=transform)
        val_dataset = load_pt_as_dataset(val_pt, transform=transform)
        test_dataset = load_pt_as_dataset(test_pt, transform=transform)

        print(f"训练集大小：{len(train_dataset)}")
        print(f"验证集大小：{len(val_dataset)}")
        print(f"测试集大小：{len(test_dataset)}")

        # === 推断维度 ===
        sample = train_dataset[0] if len(train_dataset) else (val_dataset[0] if len(val_dataset) else test_dataset[0])
        if not hasattr(sample, 'x') or sample.x is None:
            raise RuntimeError("样本缺少 x 特征，无法推断 input_dim。")
        input_dim = sample.x.size(-1)

        # 标签维度：优先认为是二分类（标量标签）；否则按整数类别或 one-hot 推断
        if hasattr(sample, 'y') and sample.y is not None:
            y = torch.as_tensor(sample.y)
            if y.dim() <= 1 and y.numel() == 1:
                output_dim = 2
            elif y.dim() == 1:
                # 粗略扫一小批样本以估计类别上界
                y_max = int(y.max().item())
                for _ in range(min(32, len(train_dataset))):
                    yi = torch.as_tensor(train_dataset[random.randint(0, len(train_dataset) - 1)].y)
                    if yi.numel() > 0:
                        y_max = max(y_max, int(yi.max().item()))
                output_dim = max(y_max + 1, 2)
            else:
                # 例如 one-hot
                output_dim = y.size(-1)
        else:
            raise RuntimeError("样本缺少 y 标签，无法推断 output_dim。")

        evaluator = SimpleBinaryEvaluator()

    else:
        raise ValueError(f"未知的数据集标识：{args.dataset}")

    # 合并后的评估加载器
    val_loader = GeoDataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = GeoDataLoader(test_dataset, batch_size=args.batch_size)

    # # ====== 全局信息 ======
    # input_dim = dataset.num_node_features
    # output_dim = dataset.num_classes
    # evaluator = Evaluator(name=args.dataset)

    # ====== 多 run 统计 ======
    run_summaries = []  # 逐次 run 的汇总（最佳指标+种子）
    auc_list, acc_list = [], []

    print(f"\n将进行 {args.runs} 次独立训练（runs）。")

    for run_idx in range(1, args.runs + 1):
        # ---------- 1) 为本次 run 确定种子 ----------
        if args.seed == -1:
            run_seed = random.randint(0, 2 ** 32 - 1)
            print(f"\n[Run {run_idx}/{args.runs}] 使用随机种子: {run_seed}")
        else:
            # 可复现的多 run：基于用户种子加偏移（让每次 run 不同、但可复现实验）
            run_seed = args.seed + (run_idx - 1)
            print(f"\n[Run {run_idx}/{args.runs}] 使用手动指定种子: {run_seed}")

        torch.manual_seed(run_seed)
        if args.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

        # ---------- 2) 本次 run 的数据加载器（含 shuffle） ----------
        train_loader = GeoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # ---------- 3) 构建模型/优化器/调度器 ----------
        model = CIGA(
            input_dim=input_dim,
            out_dim=output_dim,
            edge_dim=-1,
            emb_dim=args.emb_dim,
            num_layers=args.num_layers,
            ratio=args.ratio,
            gnn_type=args.gnn_type,
            virtual_node=bool(args.virtual_node),
            residual=bool(args.residual),
            drop_ratio=args.drop_ratio,
            JK=args.JK,
            graph_pooling=args.graph_pooling,
            c_dim=args.c_dim,
            c_in=args.c_in,
            c_rep=args.c_rep,
            c_pool=args.c_pool,
            s_rep=args.s_rep,
            sigma_len=args.sigma_len
        ).to(device)

        # Ablation controls
        model.cf_k = args.cf_k
        model.disable_budget = args.disable_budget
        model.disable_stylemix = args.disable_stylemix

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        # print("\n模型架构:")
        # print(model)
        print("\n超参数设置：")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Learning rate: {args.learning_rate}")
        print(f"  - Embedding dimension: {args.emb_dim}")
        print(f"  - GNN type: {args.gnn_type}")
        print(f"  - Dropout ratio: {args.drop_ratio}")
        print(f"  - Graph pooling: {args.graph_pooling}")
        print(f"  - Device: {args.device}")
        print(f"  - Early stop patience: {args.early_stop_patience} (0 = disabled)")

        # ---------- 4) 本次 run 的最佳指标跟踪 ----------
        '''
        best_test_auc = -1.0
        best_test_acc_at_same_epoch = 0.0
        best_val_auc_at_same_epoch = 0.0
        best_test_auc_epoch = -1
        '''
        best_val_auc = -1.0
        best_val_auc_epoch = -1
        test_auc_at_best_val_auc = 0.0
        test_acc_at_best_val_auc = 0.0
        early_stop_counter = 0

        # ---------- 5) 训练循环 ----------
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                         alpha=args.alpha, beta=args.beta, gamma=args.gamma, delta=args.delta)
            
            val_auc, val_acc = eval_metrics(model, val_loader, device, evaluator)
            test_auc, test_acc = eval_metrics(model, test_loader, device, evaluator)

            scheduler.step(val_auc)

            # ====================== 按 Val AUC 最优保存 ======================


            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_auc_epoch = epoch
                test_auc_at_best_val_auc = test_auc
                test_acc_at_best_val_auc = test_acc
                early_stop_counter = 0

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_dir,
                        f"run{run_idx}_best_val_auc_model.pt"))
            else:
                early_stop_counter += 1

            lr_cur = optimizer.param_groups[0]["lr"]
            print(f"[Run {run_idx}] Epoch [{epoch:03d}/{args.epochs}] "
                  f"TrainLoss: {train_loss:.4f}  "
                  f"ValAUC: {val_auc:.4f}  ValACC: {val_acc:.4f}  "
                  f"TestAUC: {test_auc:.4f}  TestACC: {test_acc:.4f}  "
                  f"LR: {lr_cur:.2e}")

            if (args.early_stop_patience > 0
                    and early_stop_counter >= args.early_stop_patience):
                print(f"[Run {run_idx}] 触发早停：Val AUC 连续 {early_stop_counter} 轮未提高。"
                      f"最佳 Val AUC: {best_val_auc:.4f} (epoch {best_val_auc_epoch})")
                break

        # ---------- 6) 本次 run 结束：打印小结 ----------


        print(
            "\n==> [Run {}] 最佳 Val AUC: {:.4f} (epoch {})"
            " | 对应 Test AUC: {:.4f}"
            " | 对应 Test ACC: {:.4f}".format(
                run_idx,
                best_val_auc,
                best_val_auc_epoch,
                test_auc_at_best_val_auc,
                test_acc_at_best_val_auc
            )
        )

        # 汇总验证集 AUC 最优轮对应的测试集指标
        auc_list.append(float(test_auc_at_best_val_auc))
        acc_list.append(float(test_acc_at_best_val_auc))

        run_summaries.append({
            "run": run_idx,
            "seed": int(run_seed),
            "best_val_auc": float(best_val_auc),
            "best_val_auc_epoch": int(best_val_auc_epoch),
            "test_auc_at_best_val_auc": float(test_auc_at_best_val_auc),
            "test_acc_at_best_val_auc": float(test_acc_at_best_val_auc),
        })


    # ====== 保存 summary.json（含多 run 信息） ======
    summary = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "save_dir": save_dir,
        "hyperparams": {**vars(args)},  # 会包含 runs 与最终 seed 配置
        "runs": run_summaries,  # 逐 run 的详细信息
        "multi_run_summary": {
            "auc_list": auc_list,
            "acc_list": acc_list,
        }
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n训练总结已保存到：{summary_path}")

if __name__ == '__main__':
    main()
