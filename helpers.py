import pickle, os, sys, json, lark, random, logging
sys.setrecursionlimit(100000)
from glob import glob
from lark.tree import Tree
from torch.utils.data import DataLoader, Dataset

class ProofStepData(Dataset):
    def __init__(self, opts, split):
        super().__init__()
        self.opts = opts
        self.split = split
        self.datapath = self.opts.datapath
        self.filepath = f"{self.datapath}/{self.split}"
        self.files = os.listdir(self.filepath)
        for i, file_name in enumerate(self.files):
            self.files[i] = f"{self.filepath}/{file_name}"
        random.shuffle(self.files)
        self.size = len(self.files)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return pickle.load(open(self.files[idx], "rb"))



def merge(batch):
        fields = [
            "file",
            "proof_name",
            "n_step",
            "env",
            "local_context",
            "goal",
            "is_synthetic",
            "tactic",
        ]
        data_batch = {key: [] for key in fields}
        for example in batch:
            for key, value in example.items():
                if key not in fields:
                    continue
                data_batch[key].append(value)
        return data_batch

def traverse_postorder(node, callback):
    for c in node.children:
        if isinstance(c, Tree):
            traverse_postorder(c, callback)
    callback(node)

def get_node_count_ast(ast):
    count = 0
    for tree in ast.iter_subtrees():
        count += 1
    return count
    
    
def setup_loggers(opts):
    try:
        os.remove(opts.run_log)
        os.remove(opts.res_log)
    except:
        pass
                            
    run_handler = logging.FileHandler(opts.run_log)
    res_handler = logging.FileHandler(opts.res_log)
    
    run_handler.setFormatter(logging.Formatter('%(asctime)s:\t%(message)s'))
    res_handler.setFormatter(logging.Formatter('%(asctime)s:\t%(message)s'))
    
    run_logger = logging.getLogger("run log")
    res_logger = logging.getLogger("test log")
    
    run_logger.addHandler(run_handler)
    res_logger.addHandler(res_handler)
    
    run_logger.setLevel(logging.INFO)
    res_logger.setLevel(logging.INFO)
    
    return run_logger, res_logger