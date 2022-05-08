import re
import sys
import csv
import dgl
import pickle
import pandas as pd
import networkx as nx
from collections import Counter

data_dir = './data'
mimic_3_dir = './data/mimic3'


fname = '%s/notes_labeled.csv' % mimic_3_dir
base_name = '%s/disch' % mimic_3_dir # for output
icd_file = '%s/ICD9_descriptions' % data_dir
ICD_DIAGNOSES = '%s/D_ICD_DIAGNOSES.csv' % data_dir
ICD_PROCEEURES = '%s/D_ICD_PROCEDURES.csv' % data_dir

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

code2text = {}
text2code = {}
dia_set = {}
pro_set = {}
#level_index = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
level_index = [-1] * 22334

with open(icd_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        if line[0] not in code2text.keys():
            code2text[line[0]] = line[1].replace('\n', '')
            text2code[line[1].replace('\n', '')] = line[0]


with open(ICD_DIAGNOSES, 'r') as descfile:
    r = csv.reader(descfile)
    # header
    next(r)
    for row in r:
        code = reformat(row[1], True)
        #print(code + '\t' + row[1])
        desc = row[-1]
        dia_set[code] = row[1]
        if code not in code2text.keys():
            code2text[code] = desc
            text2code[desc] = code
with open(ICD_PROCEEURES, 'r') as descfile:
    r = csv.reader(descfile)
    # header
    next(r)
    for row in r:
        code = reformat(row[1], False)
        pro_set[code] = row[1]
        # print(code + '\t' + row[1])
        desc = row[-1]
        if code not in code2text.keys():
            code2text[code] = desc
            text2code[desc] = code



code_graph_set = sorted(code2text)

idx2code = {}
code2idx = {}
for idx, code in enumerate(code_graph_set):
    idx2code[idx] = code
    code2idx[code] = idx


second_level = ['List of ICD-9 codes 001–139: infectious and parasitic diseases',
'List of ICD-9 codes 140–239: neoplasms',
'List of ICD-9 codes 240–279: endocrine, nutritional and metabolic diseases, and immunity disorders',
'List of ICD-9 codes 280–289: diseases of the blood and blood-forming organs',
'List of ICD-9 codes 290–319: mental disorders',
'List of ICD-9 codes 320–389: diseases of the nervous system and sense organs',
'List of ICD-9 codes 390–459: diseases of the circulatory system',
'List of ICD-9 codes 460–519: diseases of the respiratory system',
'List of ICD-9 codes 520–579: diseases of the digestive system',
'List of ICD-9 codes 580–629: diseases of the genitourinary system',
'List of ICD-9 codes 630–679: complications of pregnancy, childbirth, and the puerperium',
'List of ICD-9 codes 680–709: diseases of the skin and subcutaneous tissue',
'List of ICD-9 codes 710–739: diseases of the musculoskeletal system and connective tissue',
'List of ICD-9 codes 740–759: congenital anomalies',
'List of ICD-9 codes 760–779: certain conditions originating in the perinatal period',
'List of ICD-9 codes 780–799: symptoms, signs, and ill-defined conditions',
'List of ICD-9 codes 800–999: injury and poisoning']

G = dgl.DGLGraph()
G.add_nodes(len(code_graph_set))


G.add_edges(code2idx['@'], code2idx['00-99.99'])
level_index[code2idx['@']] = 0
level_index[code2idx['00-99.99']] = 1
span_code = []
for code in code_graph_set:
    if '-' in code and code != '00-99.99' and not bool(re.search('[a-zA-Z]', code)):
        code_split = code.split('-')
        if len(code_split[0]) == 2:
            G.add_edges(code2idx['00-99.99'], code2idx[code])
            span_code.append(code)
            level_index[code2idx[code]] = 2

span_id = 0
G.add_edges(code2idx['00-99.99'], code2idx['00'])
level_index[code2idx['00']] = 2
for code in code_graph_set:
    if '-' not in code and '.' not in code and len(code) == 2 and not bool(re.search('[a-zA-Z]', code)):
        span = span_code[span_id]
        start, end = span.replace('.99', '').split('-')
        if int(code) >= int(start) and int(code) <= int(end):
            G.add_edges(code2idx[span], code2idx[code])
        if int(code) == int(end):
            span_id += 1
for code in code_graph_set:
    if '-' not in code and '.' in code and not bool(re.search('[a-zA-Z]', code)):
        head = code.split('.')[0]
        if len(head) == 2:
            G.add_edges(code2idx[head], code2idx[code])


G.add_edges(code2idx['@'], code2idx['001-999.99'])

second_level_code = []
for line in second_level:
    line = line.split(':')[1][1:]
    second_level_code.append(text2code[line.upper()])
for code in second_level_code:
    G.add_edges(code2idx['001-999.99'], code2idx[code])

i = 0
third_level_code = []
G.add_edges(code2idx['290-319.99'], code2idx['290-299.99'])
G.add_edges(code2idx['290-299.99'], code2idx['290-294.99'])
G.add_edges(code2idx['290-299.99'], code2idx['295-299.99'])

third_level_code.append('290-294.99')
third_level_code.append('295-299.99')
passcode = ['001-999.99', '290-299.99', '290-294.99', '295-299.99', '800-829.99', '800-804.99', '805-809.99', '810-819.99', '820-829.99', 
            '870-897.99', '870-879.99', '880-887.99', '890-897.99']
G.add_edges(code2idx['800-999.99'], code2idx['800-829.99'])
for code in ['800-804.99', '805-809.99', '810-819.99', '820-829.99']:
    G.add_edges(code2idx['800-829.99'], code2idx[code])
    third_level_code.append(code)
G.add_edges(code2idx['800-999.99'], code2idx['870-897.99'])
for code in ['870-879.99', '880-887.99', '890-897.99']:
    G.add_edges(code2idx['870-897.99'], code2idx[code])
    third_level_code.append(code)
for code in code_graph_set:
    if '-' in code and not bool(re.search('[a-zA-Z]', code)) and code not in second_level_code:
        code_s = code.split('.')[0]
        code_split = code_s.split('-')
        if len(code_split[0]) == 3 and code not in passcode:
            start_code = code_split[0]
            end_code = code_split[1]
            second_span = second_level_code[i].split('.')[0]
            second_start, second_end = second_span.split('-')
            while(int(start_code) > int(second_end)):
                third_level_code.append(second_level_code[i])
                i += 1
                second_span = second_level_code[i].split('.')[0]
                second_start, second_end = second_span.split('-')
            if int(start_code) >= int(second_start) and int(end_code) <= int(second_end):
                G.add_edges(code2idx[second_level_code[i]], code2idx[code])
                third_level_code.append(code)
            if int(end_code) == int(second_end):
                i += 1

third_level_code.sort()
end_last = 0

i = 0
a = 0
b = 0
for code in code_graph_set:
    if '-' not in code and '.' not in code and len(code) == 3 and not bool(re.search('[a-zA-Z]', code)):
        span = third_level_code[i]
        start, end = span.split('.')[0].split('-')
        a = a + 1
        while int(code) < int(start):
            print(code)
            i += 1
            continue 
        if int(code) >= int(start) and int(code) <= int(end):
            G.add_edges(code2idx[span], code2idx[code])
            b = b + 1
        if int(code) == int(end):
            i += 1

for code in code_graph_set:
    if '-' not in code and '.' in code and not bool(re.search('[a-zA-Z]', code)):
        head = code.split('.')[0]
        if len(head) == 3:
            G.add_edges(code2idx[head], code2idx[code])


G.add_edges(code2idx['@'], code2idx['E800-E999.9'])
G.add_edges(code2idx['@'], code2idx['E000'])
G.add_edges(code2idx['@'], code2idx['E001-E030.9'])
span_code = []
span_code.append('E001-E030.9')
for code in code_graph_set:
    if bool(re.search('[E]', code)) and '-' in code and code != 'E800-E999.9' and code != 'E001-E030.9':
        G.add_edges(code2idx['E800-E999.9'], code2idx[code])
        span_code.append(code)

i = 0
G.add_edges(code2idx['E800-E999.9'], code2idx['E849'])
for code in code_graph_set:
    if bool(re.search('[E]', code)) and '-' not in code and '.' not in code:
        span = span_code[i]
        start, end = span.replace('E', '').split('.')[0].split('-')
        code_number = code.replace('E', '')
        if int(code_number) >= int(start) and int(code_number) <= int(end):
            G.add_edges(code2idx[span], code2idx[code])
        if int(code_number) == int(end):
            i += 1
for code in code_graph_set:
    if '-' not in code and '.' in code and bool(re.search('[E]', code)):
        head = code.split('.')[0]
        G.add_edges(code2idx[head], code2idx[code])

G.add_edges(code2idx['@'], code2idx['V01-V87.99'])
span_code = []
for code in code_graph_set:
    if bool(re.search('V', code)) and '-' in code and code != 'V01-V87.99':
        G.add_edges(code2idx['V01-V87.99'], code2idx[code])
        span_code.append(code)

i = 0
for code in code_graph_set:
    if bool(re.search('[V]', code)) and '-' not in code and '.' not in code:
        span = span_code[i]
        start, end = span.replace('V', '').split('.')[0].split('-')
        code_number = code.replace('V', '')
        if int(code_number) >= int(start) and int(code_number) <= int(end):
            G.add_edges(code2idx[span], code2idx[code])
        if int(code_number) == int(end):
            i = i + 1
for code in code_graph_set:
    if '-' not in code and '.' in code and bool(re.search('[V]', code)):
        head = code.split('.')[0]
        G.add_edges(code2idx[head], code2idx[code])

label_graph = {'Graph': G, 'code2text': code2text, 'text2code': text2code, 'code2idx': code2idx, 'idx2code': idx2code}
with open('./data/icd_graph_dgl.pkl', 'wb') as fw:
    pickle.dump(label_graph, fw)