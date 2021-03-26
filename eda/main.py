import os, pickle, sys, argparse
sys.path.append(os.path.abspath('../'))
from helpers import check_args

def qualid_hit_ratio(opts):
    path = opts.datapath
    file_names = os.listdir(path)
    
    hits = 0
    total_count = 0
    file_count = 0
    
    for i, file_name in enumerate(file_names):
        print(i)
        file_count += 1
        file_path = f"{path}/{file_name}"
        step = pickle.load(open(file_path, "rb"))
        
        tactic_application = step["tactic"]["text"].split(" ")
        tactic = tactic_application[0]
        
        gc = []
        lc = []
        for g in step["env"]:
            qualid = g["qualid"]
            gc.append(qualid)
        for l in step["local_context"]:
            lc.append(l["ident"])
        
        for argument in tactic_application[1:]:
            if "_" in argument and argument != "f_equal":
                total_count += 1
                for g in gc:
                    if argument in g:
                        hits += 1
                        break
                if argument in lc:
                    total_count -= 1
                        
            
    print(hits/total_count)
    print(total_count/file_count)
            
            
def convert_tactic(opts):
    path = opts.datapath
    file_names = os.listdir(path)
    
    lim = 100
    for i, file_name in enumerate(file_names):
        #print(i)
        file_path = f"{path}/{file_name}"
        step = pickle.load(open(file_path, "rb"))
                
        groups = check_args(opts, step)
        print(step["tactic"]["text"])
        print(groups)
        
        if i > lim:
            break
    
def arg_stats(opts):
    path = opts.datapath
    file_names = os.listdir(path)
    
    total = {"none": 0, "local context": 0, "global context": 0, "with clause": 0,
             "in clause": 0, "using clause": 0, "at clause": 0, "arrow clause": 0,
             "all occurrences": 0, "int": 0, "other": 0}
             
    for i, file_name in enumerate(file_names):
        print(i)
        file_path = f"{path}/{file_name}"
        step = pickle.load(open(file_path, "rb"))
                
        groups = check_args(opts.jsonpath, step)
        for k, v in groups.items():
            if v > 0:
                total[k] += 1
                
                
    freqs = {"none": 0, "local context": 0, "global context": 0, "with clause": 0,
             "in clause": 0, "using clause": 0, "at clause": 0, "arrow clause": 0,
             "all occurrences": 0, "int": 0, "other": 0}
    
    for k, v in total.items():
        freqs[k] = v/len(file_names)
        
    print(total)
    print(freqs)
            
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="EDA")
    arg_parser.add_argument("--jsonpath", type=str, default="../arg_groups/jsons")
    arg_parser.add_argument("--datapath", type=str, default="../data/train")
    arg_parser.add_argument("--function", type=str, default="convert_tactic")
    opts = arg_parser.parse_args()

    
    if opts.function == "convert_tactic":
        convert_tactic(opts)
    elif opts.function == "arg_stats":
        arg_stats(opts)
    else:
        qualid_hit_ratio()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    