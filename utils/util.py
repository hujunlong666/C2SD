import torch
import numpy as np

import random
# from texttable import Texttable

import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data

class _FromMemory(InMemoryDataset):
    """把 (data, slices) 或 list[Data] 包装成 InMemoryDataset。"""
    def __init__(self, data=None, slices=None, data_list=None, transform=None, pre_transform=None, pre_filter=None):
        self._data = data
        self._slices = slices
        self._data_list = data_list
        super().__init__('.', transform, pre_transform, pre_filter)  # root无所谓

        if data_list is not None:
            # 将 list[Data] 打包为 InMemory 结构
            self.data, self.slices = self.collate(data_list)
        else:
            # 直接使用传入的 (data, slices)
            self.data, self.slices = data, slices

    @property
    def raw_file_names(self):  # 占位
        return []

    @property
    def processed_file_names(self):  # 占位
        return []

    def download(self):
        pass

    def process(self):
        pass


def load_pt_as_dataset(pt_path, transform=None):
    """
    支持三种情况：
    1) torch.save(list_of_Data, path)
    2) torch.save((data, slices), path)  # PyG InMemoryDataset存档
    3) torch.save(single_Data, path)
    返回：torch_geometric.data.Dataset 兼容对象
    """
    obj = torch.load(pt_path, map_location='cpu')

    # 情况 1：list[Data]
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], Data)):
        return _FromMemory(data_list=obj, transform=transform)

    # 情况 3：单个 Data -> 转成 list[Data]
    if isinstance(obj, Data):
        return _FromMemory(data_list=[obj], transform=transform)

    # 情况 2：(data, slices)
    if isinstance(obj, tuple) and len(obj) == 2:
        data, slices = obj
        if isinstance(data, Data):
            return _FromMemory(data=data, slices=slices, transform=transform)

    raise ValueError(f"不认识的 .pt 内容格式：{type(obj)}，请确认保存方式。")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_partition(len_dataset, device, seed, p=[0.5,0.5]):
    '''
        group the graph randomly

        Input:   len_dataset   -> [int]
                 the number of data to be groupped
                 
                 device        -> [torch.device]
                
                 p             -> [list]
                 probabilities of the random assignment for each group
        Output: 
                 vec           -> [torch.LongTensor]
                 group assignment for each data
    '''
    assert abs(np.sum(p) - 1) < 1e-4
    
    vec = torch.tensor([]).to(device)
    for idx, idx_p in enumerate(p):
        vec = torch.cat([vec, torch.ones(int(len_dataset * idx_p)).to(device) * idx])
        
    vec = torch.cat([vec, torch.ones(len_dataset - len(vec)).to(device) * idx])
    perm = torch.randperm(len_dataset, generator=torch.Generator().manual_seed(seed))
    return vec.long()[perm]

    
def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())
    

def PrintGraph(graph):

    if graph.name:
        print("Name: %s" % graph.name)
    print("# Nodes:%6d      | # Edges:%6d |  Class: %2d" \
          % (graph.num_nodes, graph.num_edges, graph.y))

    print("# Node features: %3d| # Edge feature(s): %3d" \
          % (graph.num_node_features, graph.num_edge_features))

