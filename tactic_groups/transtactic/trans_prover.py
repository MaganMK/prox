import json, torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer, BasicTokenizer, PreTrainedTokenizer
class TransProver(nn.Module):
    def __init__(self, opts):
        super(TransProver, self).__init__()
        self.opts = opts
        with open(f'{opts.jsonpath}/tactic_groups.json', 'r') as f:
            self.tactic_groups = json.load(f)
        with open(f'{opts.jsonpath}/tactic_groups_reverse.json', 'r') as f: 
            self.tactic_groups_reverse = json.load(f)
        
  
                                 
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.config = BertConfig(hidden_dropout_prob=self.opts.dropout, 
                                 attention_probs_dropout_prob=self.opts.dropout,
                                 num_labels = len(self.tactic_groups),
                                 num_hidden_layers=self.opts.num_hidden,
                                 num_attention_heads=self.opts.num_attention,
                                 vocab_size = len(self.tokenizer))
                                 
        self.bert = BertForSequenceClassification(config=self.config)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, batch):
        
        if self.opts.sexpression:
            goal_texts = [goal["sexpression"] for goal in batch["goal"]]
        else:
            goal_texts = [goal['text'] for goal in batch['goal']]
        
        for i, txt in enumerate(goal_texts):
            if txt == None:
                goal_texts[i] = "None"
                
        
        encodings = self.tokenizer.batch_encode_plus(goal_texts,
                                                     add_special_tokens=True,
                                                     return_attention_mask=True,
                                                     padding='max_length',
                                                     return_tensors='pt',
                                                     truncation=True,
                                                     max_length=self.opts.tokenizer_length)
        """
        
        for goal_text in goal_texts:
            print(len(self.tokenizer))
            print(goal_text)
            
            tokens = self.tokenizer.tokenize(goal_text)
            print(tokens)
            
            encoding = self.tokenizer(goal_text,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               padding='max_length',
                                               return_tensors='pt',
                                               truncation=True,
                                               max_length=self.opts.tokenizer_length)
            print(encoding)
        
        """
                                                     
        tokens = encodings["input_ids"].to(self.opts.device)
        attention_masks = encodings["attention_mask"].to(self.opts.device)
        
        true_tactics = [tactic['text'] for tactic in batch['tactic']]
        groups_true = self.get_groups(true_tactics)
        labels = self.tactic_space_mapping(groups_true, len(goal_texts))
    
        input = {"input_ids": tokens, "attention_mask": attention_masks, "labels": labels}
    
        output = self.bert(**input, output_hidden_states=True, output_attentions=True)
        preds = self.softmax(output.logits)
        print(preds)
        
        return self.get_groups_preds(preds), groups_true, output.loss
    
    
    def get_groups(self, tactics):
        res = []
        for tactic in tactics:
            all_actions = tactic.split(" ")
            if all_actions[0] in self.tactic_groups_reverse:
                res.append(self.tactic_groups_reverse[all_actions[0]])
            else:
                res.append("goal break up/other")
        return res

    def tactic_space_mapping(self, actions, current_batchsize):
        target = torch.empty(current_batchsize, dtype=torch.long).to(self.opts.device)
        for i, action in enumerate(actions):
            index = list(self.tactic_groups.keys()).index("goal break up/other")
            for group in self.tactic_groups.keys():
                if group == action:
                    index = list(self.tactic_groups.keys()).index(group)
            target[i] = index
        return target

    def get_groups_preds(self, preds):
        res = []
        for pred in preds:
            current_pred = list(self.tactic_groups.keys())[torch.argmax(pred)]
            res.append(current_pred)
        return res