import multiprocessing, sys, json, logging, gc, os, argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.abspath('../'))
#sys.path.append(os.path.abspath('./'))
from helpers import ProofStepData, merge, setup_loggers, build_csv
from ffn.ffn_prover import FFNProver
from gast.gast_prover import GASTProver
from transtactic.trans_prover import TransProver


def train(opts):
    
    run_logger, res_logger = setup_loggers(opts)
    
    if str(opts.device) == "cpu":
        run_logger.info(f"using CPU with {multiprocessing.cpu_count()} cores")
        run_logger.info(f"torch uses {torch.get_num_threads()} theards")
        run_logger.info(f"max recurssion is {sys.getrecursionlimit()}")
    else:
        run_logger.info("Using GPU")
        run_logger.info(f"torch uses {torch.get_num_threads()} theards")
        run_logger.info(f"max recurssion is {sys.getrecursionlimit()}")

    res_logger.info(f"opts -> {opts}")
    
    start_time = datetime.now()

    train = DataLoader(ProofStepData(opts, "train"), opts.batchsize, collate_fn=merge, num_workers = opts.num_workers)
    valid = DataLoader(ProofStepData(opts, "valid"), opts.batchsize, collate_fn=merge, num_workers = opts.num_workers)
    res_logger.info(f"training size -> {len(train)}")
    res_logger.info(f"valid size -> {len(valid)}")

    if opts.prover == "gast":
        model = GASTProver(opts)
    elif opts.prover == "ffn":
        model = FFNProver(opts)
    elif opts.prover == "trans":
        model = TransProver(opts)
        
    model.to(opts.device)
    res_logger.info(model)
    
    if opts.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.l2)
    if opts.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=opts.lr_reduce_patience, verbose=True)


    for n in range(opts.epochs):
        run_logger.info(f"epoch: {n}")
        res_logger.info(f"epoch: {n}")

        # training
        loss_avg_train = 0
        num_correct_train = 0
        pred_freq_train = {}
        model.train()
        
        for i, data_batch in enumerate(train):
            preds, true, loss = model(data_batch)
            loss_avg_train += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            gc.collect()
            
            elapsed_time = datetime.now() - start_time
            run_logger.info(f"{i}/{len(train)} -> {100*(i/len(train))}% ({elapsed_time})")
            
            for j in range(len(preds)):
                if preds[j] == true[j]:
                    num_correct_train += 1
                pred_freq_train[preds[j]] = pred_freq_train.get(preds[j], 0) + 1
                
        loss_avg_train /= len(train)
        acc_train = num_correct_train/121644
        
        # validation
        run_logger.info("validation...")
        model.eval()
        loss_avg_valid = 0
        num_correct_valid = 0
        pred_freq_valid = {}

        for i, data_batch in enumerate(valid):
            preds, true, loss = model(data_batch)
            loss_avg_valid += loss.item()

            for j in range(len(preds)):
                if preds[j] == true[j]:
                    num_correct_valid += 1
                pred_freq_valid[preds[j]] = pred_freq_valid.get(preds[j], 0) + 1
                          
            elapsed_time = datetime.now() - start_time
            run_logger.info(f"{i}/{len(valid)} -> {100*(i/len(valid))}% ({elapsed_time})")
        
        loss_avg_valid /= len(valid)
        acc_valid = num_correct_valid/68180
    
        res_logger.info("###################")
        res_logger.info(f"train guesses: {pred_freq_train}")
        res_logger.info(f"validation guesses: {pred_freq_valid}")
        res_logger.info(f"train losses: {loss_avg_train}")
        res_logger.info(f"validation losses: {loss_avg_valid}")
        res_logger.info(f"train accuracy: {acc_train}")
        res_logger.info(f"validation accuracy: {acc_valid}")
        res_logger.info("###################")
        build_csv(opts, loss_avg_train, loss_avg_valid, acc_train, acc_valid)
        
        
        scheduler.step(loss_avg_valid)
        



def sanity_check(opts):
    train = DataLoader(ProofStepData(opts, "train"), opts.batchsize, collate_fn=merge, num_workers = opts.num_workers)
    valid = DataLoader(ProofStepData(opts, "valid"), opts.batchsize, collate_fn=merge, num_workers = opts.num_workers)

    if opts.prover == "gast":
        model = GASTProver(opts)
    elif opts.prover == "ffn":
        model = FFNProver(opts)
    elif opts.prover == "trans":
        model = TransProver(opts)
    
    model.to(opts.device)

    if opts.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.l2)
    if opts.optimizer == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=opts.lr_reduce_patience, verbose=True)
    
    for n in range(opts.epochs):
        
        # training
        loss_avg_train = 0
        num_correct_train = 0
        pred_freq_train = {}
        model.train()
        counter = 0
        proof_step_counter = 0
        for i, data_batch in enumerate(train):
            counter += 1
            preds, true, loss = model(data_batch)
            loss_avg_train += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            gc.collect()
            
            for j in range(len(preds)):
                proof_step_counter += 1
                if preds[j] == true[j]:
                    num_correct_train += 1
                pred_freq_train[preds[j]] = pred_freq_train.get(preds[j], 0) + 1
                
            if counter > opts.sanity_batches:
                break
                
        loss_avg_train /= counter
        acc_train = num_correct_train/proof_step_counter
        
        print(f"###### epoch: {n} #######")
        print(f"train guesses: {pred_freq_train}")
        print(f"train losses: {loss_avg_train}")
        print(f"train accuracy: {acc_train}")
        print("###################")
    
        
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="train gast models")
    
    # paths
    arg_parser.add_argument("--datapath", type=str, default="../data/")
    arg_parser.add_argument("--jsonpath", type=str, default="./jsons")
    arg_parser.add_argument("--run_log", type=str, default="./logs/run.log")
    arg_parser.add_argument("--res_log", type=str, default="./logs/res.log")
    arg_parser.add_argument("--res_csv", type=str, default="./logs/res.csv")
    arg_parser.add_argument("--sanity", type=bool, default=False)
    arg_parser.add_argument("--sanity_batches", type=int, default=4)
    arg_parser.add_argument("--prover", type=str, default="gast")
    
    # general/optimization
    arg_parser.add_argument("--num_workers", type=int, default=2)
    arg_parser.add_argument("--epochs", type=int, default=100)
    arg_parser.add_argument("--batchsize", type=int, default=16)
    arg_parser.add_argument("--embedding_info", type=str, default="goal")
    arg_parser.add_argument("--optimizer", type=str, default="adam")
    arg_parser.add_argument("--scheduler", type=str, default="plateau")
    
    arg_parser.add_argument("--dropout", type=float, default=0.1)
    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument("--l2", type=float, default=0.000001)
    arg_parser.add_argument("--lr_reduce_patience", type=float, default=3)
    
    # gast
    arg_parser.add_argument("--embedding_dim", type=int, default=256)
    arg_parser.add_argument("--embedder", type=str, default="sageconv")
    arg_parser.add_argument("--pooling", type=str, default="mean")
    arg_parser.add_argument("--node_pooling", type=str, default="none")
    arg_parser.add_argument("--norm", type=str, default="none")
    arg_parser.add_argument("--predictor", type=str, default="linear")
    arg_parser.add_argument("--num_message_layers", type=int, default=1)
    arg_parser.add_argument("--hops", type=int, default=1)
    
    # transtactic
    arg_parser.add_argument("--sexpression", type=bool, default=False)
    arg_parser.add_argument("--tokenizer_length", type=int, default=256)
    arg_parser.add_argument("--num_hidden", type=int, default=12)
    arg_parser.add_argument("--num_attention", type=int, default=12)
    arg_parser.add_argument("--vocab_size", type=int, default=30522)
    
    opts = arg_parser.parse_args()
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if opts.sanity:
        sanity_check(opts)
    else:
        train(opts)
        