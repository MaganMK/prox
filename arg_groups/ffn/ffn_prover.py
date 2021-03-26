import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from helpers import ProofStepData, merge, traverse_postorder, get_node_count_ast, check_args_batch
import json

class FFNProver(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.embedder = opts.embedder
        with open(f'{opts.jsonpath}/arg_groups.json', 'r') as f: 
            self.arg_groups = json.load(f)
        with open(f'{opts.jsonpath}/arg_groups_reverse.json', 'r') as f: 
            self.arg_groups_reverse = json.load(f)
        with open(f'{opts.jsonpath}/nonterminals.json', 'r') as f: 
            self.nonterminals = json.load(f)
        
        self.ffn = nn.Sequential(
            nn.Linear(len(self.nonterminals), len(self.nonterminals)),
            nn.ReLU(),
            nn.Dropout(p=self.opts.dropout),
            nn.Linear(len(self.nonterminals), len(self.nonterminals)),
            nn.ReLU(),
            nn.Dropout(p=self.opts.dropout),
            nn.Linear(len(self.nonterminals), len(self.arg_groups)),
            nn.Dropout(p=self.opts.dropout),
        )
        self.activation = nn.Sigmoid()

    def forward(self, batch):
        # compute goal embeddings
        goal_asts = [goal['ast'] for goal in batch['goal']]
        goal_encodings = self.ast_encodings(goal_asts)

        preds = self.ffn(goal_encodings)
        
        true_groups = self.get_groups(batch)
        loss = self.compute_loss(preds, true_groups, len(goal_asts))
        
        
        preds = self.activation(preds)
        pred_groups = self.get_groups_preds(preds)
        return pred_groups, true_groups, loss
        
    
    def ast_encode(self, ast):
        res = [0.0]*len(self.nonterminals)
        
        def callbck(node):
            index = self.nonterminals.index(node.data)
            res[index] += 1.0

        traverse_postorder(ast, callbck)
        return torch.tensor(res).to(self.opts.device)
        
    def ast_encodings(self, asts):
        encodings = []
        for i, ast in enumerate(asts):
            if ast != None:
                encoding = self.ast_encode(ast)
                encodings.append(encoding)
        return torch.stack(encodings).to(self.opts.device)
                
    def compute_loss(self, groups_pred, groups_true, current_batchsize):
        targets = self.tactic_space_mapping(groups_true, current_batchsize)
        criterion = nn.BCEWithLogitsLoss().to(self.opts.device)
        loss = criterion(groups_pred, targets)
        return loss

    def get_groups(self, batch):
        res = []
        all_groups = check_args_batch("./jsons", batch)
        for step in all_groups:
            tmp = []
            for group, v in step.items():
                if v >= 1:
                    tmp.append(group)
            res.append(tmp)
        return res
    
    def get_groups_preds(self, preds):
        res = []
        for pred in preds:
            tmp = []
            for i, p in enumerate(pred):
                if p >= 0.5:
                    current_pred = list(self.arg_groups.keys())[i]
                    tmp.append(current_pred)
            res.append(tmp)
        return res
    
    def tactic_space_mapping(self, actions, current_batchsize):
        target = torch.zeros([current_batchsize, len(self.arg_groups)], dtype=torch.float).to(self.opts.device)
        for i, action in enumerate(actions):
            for group in action:
                index = list(self.arg_groups.keys()).index(group)
                target[i][index] = 1
        return target


