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
    
    
def build_csv(opts, train_loss, valid_loss, train_acc, valid_acc):
    path = opts.res_csv
    f = open(path, 'a')
    f.write(f"{train_loss},{valid_loss},{train_acc},{valid_acc}\n")
    f.close()
    
    
def check_args(json_path, step):
    tactic_app = step["tactic"]["text"].split(" ")
    tactic = tactic_app[0]
    unprepped_args = tactic_app[1:]
    args = []
    for arg in unprepped_args:
        if "," in arg:
            arg = arg.split(",")
            for a in arg:
                if a != '':
                     args.append(a)
        else:
            args.append(arg)  
    gc = []
    lc = []
    for g in step["env"]:
        qualid = g["qualid"]
        gc.append(qualid)
    for l in step["local_context"]:
        lc.append(l["ident"])
    
    with open(f"{json_path}/arg_groups.json") as f:
        arg_groups = json.load(f)
    with open(f"{json_path}/arg_groups_reverse.json") as f:
        arg_groups_reverse = json.load(f)
    
    res = {}
    for k, v in arg_groups.items():
        res[k] = 0
    
    for arg in args:
        if arg in lc:
            res["local context"] += 1
        elif ("." in arg or "_" in arg) and arg != "f_equal":
            res["global context"] += 1
        elif arg in arg_groups_reverse.keys():
            key = arg_groups_reverse[arg]
            res[key] = 1
        elif type(arg) == int:
            res["int"] += 1
        elif "," not in arg:
            res["global context"] += 1
        else:
            print(tactic_app)
            print(arg)
            res["other"] += 1
            
    count = 0
    for e in res.values():
        if e == 0:
            count += 1
    if count == len(res):
        res["none"] = 1
    
    return res
    
def check_args_batch(json_path, batch):
    def check(jsonpath, tactic_app, gc, lc):
        tactic_app = tactic_app["text"].split(" ")
        tactic = tactic_app[0]
        unprepped_args = tactic_app[1:]
        args = []
        for arg in unprepped_args:
            if "," in arg:
                arg = arg.split(",")
                for a in arg:
                    if a != '':
                        args.append(a)
            else:
                args.append(arg)
                
        with open(f"{jsonpath}/arg_groups.json") as f:
            arg_groups = json.load(f)
        with open(f"{jsonpath}/arg_groups_reverse.json") as f:
            arg_groups_reverse = json.load(f)
    
        res = {}
        for k, v in arg_groups.items():
            res[k] = 0
    
        for arg in args:
            if arg in lc:
                res["local context"] += 1
            elif ("." in arg or "_" in arg) and arg != "f_equal":
                res["global context"] += 1
            elif arg in arg_groups_reverse.keys():
                key = arg_groups_reverse[arg]
                res[key] = 1
            elif type(arg) == int:
                res["int"] += 1
            elif "," not in arg:
                res["global context"] += 1
            else:
                print(tactic_app)
                print(arg)
                res["other"] += 1
        
        count = 0
        for e in res.values():
            if e == 0:
                count += 1
        if count == len(res):
            res["none"] = 1
        return res
    
    tactic_apps = batch["tactic"]
    lcs = batch["local_context"]
    gcs = batch["env"]
    
    out = []
    for i in range(len(tactic_apps)):
        gc = []
        lc = []
        for g in gcs[i]:
            qualid = g["qualid"]
            gc.append(qualid)
        for l in lcs[i]:
            lc.append(l["ident"])
        
        tactic_app = tactic_apps[i]
        
        tmp = check(json_path, tactic_app, gc, lc)
        out.append(tmp)
    
    return out
        
        
        
        
                 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    