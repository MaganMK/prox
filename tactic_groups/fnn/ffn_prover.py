import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F
from helpers import ProofStepData, merge, traverse_postorder, get_node_count_ast
import json

class FFNProver(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.embedder = opts.embedder
        with open(f'{opts.jsonpath}/tactic_groups.json', 'r') as f: 
            self.tactic_groups = json.load(f)
        with open(f'{opts.jsonpath}/tactic_groups_reverse.json', 'r') as f: 
            self.tactic_groups_reverse = json.load(f)
        with open(f'{opts.jsonpath}/nonterminals.json', 'r') as f: 
            self.nonterminals = json.load(f)
        
        self.ffn = nn.Sequential(
            nn.Linear(len(self.nonterminals), len(self.nonterminals)),
            nn.ReLU(),
            nn.Dropout(p=self.opts.dropout),
            nn.Linear(len(self.nonterminals), len(self.nonterminals)),
            nn.ReLU(),
            nn.Dropout(p=self.opts.dropout),
            nn.Linear(len(self.nonterminals), len(self.tactic_groups)),
            nn.ReLU(),
            nn.Dropout(p=self.opts.dropout),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        # compute goal embeddings
        goal_asts = [goal['ast'] for goal in batch['goal']]
        goal_encodings = self.ast_encodings(goal_asts)

        preds = self.ffn(goal_encodings)
        

        true_tactics = [tactic['text'] for tactic in batch['tactic']]
        true_groups = self.get_groups(true_tactics)
        loss = self.compute_loss(preds, true_groups, len(true_tactics))
        
        preds = self.softmax(preds)
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
                

    def get_groups(self, tactics):
        res = []
        for tactic in tactics:
            all_actions = tactic.split(" ")
            if all_actions[0] in self.tactic_groups_reverse:
                res.append(self.tactic_groups_reverse[all_actions[0]])
            else:
                res.append("goal break up/other")
        return res
    
    def get_groups_preds(self, preds):
        res = []
        for pred in preds:
            current_pred = list(self.tactic_groups.keys())[torch.argmax(pred)]
            res.append(current_pred)
        return res
    
    def compute_loss(self, groups_pred, groups_true, current_batchsize):
        targets = self.tactic_space_mapping(groups_true, current_batchsize)
        criterion = nn.CrossEntropyLoss().to(self.opts.device)
        loss = criterion(groups_pred, targets)
        return loss

    def tactic_space_mapping(self, actions, current_batchsize):
        target = torch.empty(current_batchsize, dtype=torch.long).to(self.opts.device)
        for i, action in enumerate(actions):
            index = list(self.tactic_groups.keys()).index("goal break up/other")
            for group in self.tactic_groups.keys():
                if group == action:
                    index = list(self.tactic_groups.keys()).index(group)
            target[i] = index
        return target
