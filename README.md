# Package Dependencies

* allennlp == 0.8.4
* gensim == 3.8.3
* pytorch==1.7.1
* spacy == 2.1.9
* nltk == 3.6.1
* python == 3.8.8
* pytorch-pretrained-bert == 0.6.2
* transformers == 4.9.2
* scikit-learn == 0.24.1
* dgl == 0.6.1
* torch-scatter == 2.0.6
* torch-sparse == 0.6.9
* torch-geometric == 1.7.0

You can use the following command (recommended):
~~~
pip install -r requirements.txt
~~~

## Preprossing 

### Clinical Document

The data can be downloaded from [MIMIC-III](https://mimic.mit.edu/). The structure of data files can be shown like:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions.txt
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
The hierarchical ICD graph can be obtained by running ```python icd_graph.py```.

The corresponding ICD code file can be obtained by running ```python preprocess_mimic3.py```.

To obtain the clause interaction graph, firstly run the ```python preprocess_mimic3_raw.py``` to obtain raw discharge summaries. And then run the ```python clause.py``` to get clause clustering results. Finally, run ```python code_matching.py``` to obtain interaction connection.


## Training
MIMIC-III-50
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 200 --batch_size 4 --model GRU --lr 1e-4 --clause True --gpu 1 --bidirectional --criterion prec_at_5
~~~
MIMIC-III-Full
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 200 --batch_size 4 --model GRU --lr 1e-4 --clause True --gpu 1 --bidirectional --criterion prec_at_8 --Y full --data_path ./data/mimic3/train_full.csv
~~~

## Testing
MIMIC-III-50
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 200 --batch_size 4 --model GRU --lr 1e-4 --clause True --gpu 1 --bidirectional --test_model=save_model_path
~~~
MIMIC-III-Full
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 200 --batch_size 4 --model GRU --lr 1e-4 --clause True --gpu 1 --bidirectional --Y full --data_path ./data/mimic3/train_full.csv --test_model=save_model_path
~~~

