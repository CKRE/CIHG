import json
import pickle
import dgl
import time
import numpy as np
import networkx as nx
import pandas as pd
import pytorch_lightning as pl
from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import fuzz
from tqdm import tqdm
from multiprocessing import Pool
from dataloader import load_lookups
from options import args

tokenize = RegexpTokenizer(r'\w+')

def find_max_matching(line):
    text = line['text']
    text = [t.lower() for t in tokenize.tokenize(text) if not t.isnumeric()]
    text = ' '.join(text)
    fuzzy_matching_score = []
    for code_text in code_text_set_lower:
        fuzzy_matching_score.append(fuzz.partial_ratio(code_text.lower(), text.lower()))
    max_score = max(fuzzy_matching_score)
    max_score_index = fuzzy_matching_score.index(max(fuzzy_matching_score))
    code = text2code[code_text_set[max_score_index]]
    line['match_code'] = code
    line['match_score'] = max_score
    return line



if __name__ == "__main__":
    args.Y = 'full'
    clause_file = './data/mimic3/clauses_full.json'

    with open('./data/icd_graph_dgl.pkl', 'rb') as f:
        label_processed = pickle.load(f)
        label_graph = label_processed['Graph']
        code2text = label_processed['code2text']
        text2code = label_processed['text2code']
        code_text_set = list(text2code.keys())

    dicts = load_lookups(args, False)
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    code_list = list(c2ind.keys())
    code_text_set = [code2text[code] for code in code_list]
    code_text_set_lower = []
    for code in code_text_set:
        code = [t.lower() for t in tokenize.tokenize(code) if not t.isnumeric()]
        code = ' '.join(code)
        code_text_set_lower.append(code)
    
    with open(clause_file, 'r', encoding='utf-8') as f:
        clause_result = json.load(f)
    #clause_result = clause_result[:10000]
    #for i in clause_result:
    #    find_max_matching(i)

    with Pool(10) as p:
        matching_result = p.map(find_max_matching, clause_result)
    with open('./data/mimic3/matching_full.json', 'w') as fp:
        json.dump(matching_result, fp)
