import enum
import re
import sys
import csv
import json
import pickle
import networkx
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from SentenceClustering import SentenceClustering

mimic_3_dir = './data/mimic3'
data_dir = './data'

train_filename = '%s/train_full_raw.csv' % (mimic_3_dir)
dev_filename = '%s/dev_full_raw.csv' % (mimic_3_dir)
test_filename = '%s/test_full_raw.csv' % (mimic_3_dir)
print(train_filename)
print(dev_filename)
print(test_filename)
clauses = []

with open(train_filename, 'r') as f:
    f_reader = csv.reader(f)
    next(f_reader)
    for j, line in enumerate(tqdm(f_reader)):

        if j>100:
            break

        part_num = 0
        paragraph_num = 0
        sentence_num = 0
        clauses_num = 0
        
        document = line[2]
        parts = document.split('\n\n\n')
        for part in parts:
            paragraphs = part.split('\n\n')
            for paragraph in paragraphs:
                paragraph = paragraph.replace('\n', ' ')
                paragraph = sent_tokenize(paragraph)
                
                for sent in paragraph:
                    if sent != '.':
                        sents = re.split(r',|;', sent)
                        for sentence in sents:
                            if sentence != '' and sentence != ' ':
                                sentence = {'text': sentence, 'ha': line[1], 'cl': clauses_num, 'se': sentence_num, 'set': 'train'}
                                clauses.append(sentence)
                                clauses_num += 1
                        sentence_num += 1
                paragraph_num += 1
            part_num += 1
print(len(clauses))


with open(dev_filename, 'r') as f:
    f_reader = csv.reader(f)
    next(f_reader)
    for j, line in enumerate(tqdm(f_reader)):

        if j>100:
            break

        part_num = 0
        paragraph_num = 0
        sentence_num = 0
        clauses_num = 0
        
        document = line[2]
        parts = document.split('\n\n\n')
        for part in parts:
            paragraphs = part.split('\n\n')
            for paragraph in paragraphs:
                paragraph = paragraph.replace('\n', ' ')
                paragraph = sent_tokenize(paragraph)

                for sent in paragraph:
                    if sent != '.':
                        sents = re.split(r',|;', sent)
                        for sentence in sents:
                            if sentence != '' and sentence != ' ':
                                sentence = {'text': sentence, 'ha': line[1], 'cl': clauses_num, 'se': sentence_num, 'set': 'train'}
                                clauses.append(sentence)
                                clauses_num += 1
                        sentence_num += 1
                paragraph_num += 1
            part_num += 1
print(len(clauses))


with open(test_filename, 'r') as f:
    f_reader = csv.reader(f)
    next(f_reader)
    for j, line in enumerate(tqdm(f_reader)):

        if j>100:
            break

        part_num = 0
        paragraph_num = 0
        sentence_num = 0
        clauses_num = 0
        
        document = line[2]
        parts = document.split('\n\n\n')
        for part in parts:
            paragraphs = part.split('\n\n')
            for paragraph in paragraphs:
                paragraph = paragraph.replace('\n', ' ')
                paragraph = sent_tokenize(paragraph)

                for sent in paragraph:
                    if sent != '.':
                        sents = re.split(r',|;', sent)
                        for sentence in sents:
                            if sentence != '' and sentence != ' ':
                                sentence = {'text': sentence, 'ha': line[1], 'cl': clauses_num, 'se': sentence_num, 'set': 'train'}
                                clauses.append(sentence)
                                clauses_num += 1
                        sentence_num += 1
                paragraph_num += 1
            part_num += 1
print(len(clauses))


sents = [clause['text'] for clause in clauses]
nclusters = 20
sent_clus = SentenceClustering(sents=sents, nclusters=nclusters, visualization=False)
clusters = sent_clus.kmeans_clustering()

for cluster in range(len(clusters.keys())):
    for i, sentence in enumerate(clusters[cluster]):
        clauses[sentence]['class'] = cluster

with open('%s/clauses_full.json' % (mimic_3_dir), 'w') as fp:
    json.dump(clauses, fp)
print('It has been saved in clauses_full.json')