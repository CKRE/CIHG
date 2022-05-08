
from options import args_parser
import random
import numpy as np
import torch
import csv
import sys
from dataloader import load_lookups, prepare_instance, prepare_instance_bert, MyDataset, my_collate, my_collate_bert, prepare_instance_Clause, my_collate_clause
from utils import early_stop, save_everything
from models import pick_model
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import time
import json
import pickle
from train_test import train, test, train_Clause, test_Clause
from transformers import AdamW, get_linear_schedule_with_warmup
from nltk.tokenize import RegexpTokenizer


if __name__ == "__main__":
    args = args_parser()
    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    print(args)
    
    if sys.platform == 'win32':
        csv.field_size_limit(2147483647)
    else:    
        csv.field_size_limit(sys.maxsize)

    if args.Y == 'full':
        args.tune_wordemb = True
    else:
        args.tune_wordemb = False

    # load vocab and other lookups
    print("loading lookups...")
    desc_embed = args.lmbda > 0
    
    dicts = load_lookups(args, desc_embed=desc_embed)
    model = pick_model(args, dicts)

    if not args.test_model:
        if not args.weight_tuning:
            if args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
            if args.optimizer == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, betas=(0.9, 0.999))
        else:
            optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=0.9)
    else:
        optimizer = None

    if args.model != 'elmo' and args.tune_wordemb == False:
        model.freeze_net()

    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    if args.model.find("bert") != -1:
        prepare_instance_func = prepare_instance_bert
    else:
        if args.clause:
            prepare_instance_func = prepare_instance_Clause
        else:
            prepare_instance_func = prepare_instance
    
    if args.clause:
        clause_file = './data/mimic3/matching_full.json'
        with open(clause_file, 'r', encoding='utf-8') as f:
            clause_result = json.load(f)
        clause_dataset = {}
        for line in clause_result:
            hadm_id = line['ha']
            if hadm_id not in clause_dataset.keys():
                clause_dataset[hadm_id] = []
            clause_dataset[hadm_id].append(line)
        icd_file = "./data/icd_graph_dgl.pkl"
        with open(icd_file, "rb") as f:
            icd_data = pickle.load(f)

        tokenizer = RegexpTokenizer(r'\w+')
        w2ind = dicts['w2ind']
        
        label_graph = icd_data['Graph']
        icd_code2text = icd_data['code2text']
        icd_text2code = icd_data['text2code']
        code_text = [icd_code2text[key] for key in icd_code2text]
        icd_description_length = 20
        icd_description_inputs = []
        for icd_description in code_text:
            icd_description = [t.lower() for t in tokenizer.tokenize(icd_description) if not t.isnumeric()]
            icd_description = " ".join(icd_description)
            icd_description_ids = [int(w2ind[w]) if w in w2ind else len(w2ind) + 1 for w in icd_description.split()]
            icd_description_ids = icd_description_ids[:icd_description_length]
            icd_description_ids = icd_description_ids + [0] * (icd_description_length - len(icd_description_ids))
            icd_description_inputs.append(icd_description_ids)

        train_instances = prepare_instance_func(dicts, args.data_path, args, args.MAX_LENGTH, clause_dataset, icd_data)
        print("train_instances {}".format(len(train_instances)))
        if args.version != 'mimic2':
            dev_instances = prepare_instance_func(dicts, args.data_path.replace('train','dev'), args, args.MAX_LENGTH, clause_dataset, icd_data)
            print("dev_instances {}".format(len(dev_instances)))
        else:
            dev_instances = None
        test_instances = prepare_instance_func(dicts, args.data_path.replace('train','test'), args, args.MAX_LENGTH, clause_dataset, icd_data)
        print("test_instances {}".format(len(test_instances)))
    else:
        train_instances = prepare_instance_func(dicts, args.data_path, args, args.MAX_LENGTH)
        print("train_instances {}".format(len(train_instances)))
        if args.version != 'mimic2':
            dev_instances = prepare_instance_func(dicts, args.data_path.replace('train','dev'), args, args.MAX_LENGTH)
            print("dev_instances {}".format(len(dev_instances)))
        else:
            dev_instances = None
        test_instances = prepare_instance_func(dicts, args.data_path.replace('train','test'), args, args.MAX_LENGTH)
        print("test_instances {}".format(len(test_instances)))

    if args.model.find("bert") != -1:
        collate_func = my_collate_bert
    else:
        if args.clause:
            collate_func = my_collate_clause
        else:
            collate_func = my_collate

    if args.clause:
        train_loader = DataLoader(MyDataset(args, train_instances, icd_description_inputs), args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
        if args.version != 'mimic2':
            dev_loader = DataLoader(MyDataset(args, dev_instances, icd_description_inputs), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
        else:
            dev_loader = None
        test_loader = DataLoader(MyDataset(args, test_instances, icd_description_inputs), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
    else:
        train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
        if args.version != 'mimic2':
            dev_loader = DataLoader(MyDataset(dev_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)
        else:
            dev_loader = None
        test_loader = DataLoader(MyDataset(test_instances), 1, shuffle=False, collate_fn=collate_func, num_workers=args.num_workers, pin_memory=False)


    if not args.test_model and args.model.find("bert") != -1:
        if args.use_lr_layer_decay:
            optimizer = AdamW(
                [{'params':model.bert.embeddings.parameters(), 'lr': args.lr * args.lr_layer_decay ** (len(model.bert.encoder.layer) + 2)}]
                +
                [{'params': module_list_item.parameters(), 'lr': args.lr * args.lr_layer_decay ** (len(model.bert.encoder.layer) + 1 - index), } 
                for index, module_list_item in enumerate(model.bert.encoder.layer)]
                +
                [{'params':model.bert.pooler.parameters(), 'lr': args.lr * args.lr_layer_decay ** 1}]
                +
                [{'params':model.final.parameters(), 'lr': args.lr * args.lr_layer_decay ** 0}]
            , correct_bias = False, weight_decay = args.weight_decay)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False, weight_decay=args.weight_decay)

    if not args.test_model and args.use_lr_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.warm_up * len(train_loader) * args.n_epochs, (1-args.warm_up) * len(train_loader) * args.n_epochs)
    else:
        lr_scheduler = None

    test_only = args.test_model is not None

    for epoch in range(args.n_epochs):
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(args.MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H_%M_%S', time.localtime())]))
            os.makedirs(model_dir)
        elif args.test_model:
            model_dir = os.path.dirname(os.path.abspath(args.test_model))

        if not test_only:
            if args.debug:
                for param_group in optimizer.param_groups:
                    print("Learning rate: % .10f" % (param_group['lr']))
            
            epoch_start = time.time()
            if args.clause:
                losses = train_Clause(args, model, optimizer, epoch, args.gpu, train_loader, lr_scheduler)
            else:
                losses = train(args, model, optimizer, epoch, args.gpu, train_loader, lr_scheduler)
            loss = np.mean(losses)
            epoch_finish = time.time()
            print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
        else:
            loss = np.nan

        fold = 'test' if args.version == 'mimic2' else 'dev'
        dev_instances = test_instances if args.version == 'mimic2' else dev_instances
        dev_loader = test_loader if args.version == 'mimic2' else dev_loader
        if epoch == args.n_epochs - 1:
            print("last epoch: testing on dev and test sets")
            test_only = True

        # test on dev
        evaluation_start = time.time()
        if args.clause:
            metrics = test_Clause(args, model, args.data_path, fold, args.gpu, dicts, dev_loader)
        else:
            metrics = test(args, model, args.data_path, fold, args.gpu, dicts, dev_loader)
        evaluation_finish = time.time()
        print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
        if test_only or epoch == args.n_epochs - 1:
            if args.clause:
                metrics_te = test_Clause(args, model, args.data_path, "test", args.gpu, dicts, test_loader)
            else:
                metrics_te = test(args, model, args.data_path, "test", args.gpu, dicts, test_loader)
        else:
            metrics_te = defaultdict(float)

        metrics_tr = {'loss': loss}
        metrics_all = (metrics, metrics_te, metrics_tr)

        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)
        save_everything(args, metrics_hist_all, model, model_dir, None, args.criterion, test_only)
        sys.stdout.flush()

        if test_only:
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                args.test_model = '%s/model_best_%s.pth' % (model_dir, args.criterion)
                model = pick_model(args, dicts)
