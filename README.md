# Soda-ClariQ

This is the code repository of our team (named "Soda") for the challenge [ConvAI3: Clarifying Questions for Open-Domain Dialogue Systems (ClariQ)](http://convai.io/). Our model adopts "BERT-based" methods for both clarification prediction and question ranking.

## Requirements
Our experiments are based on Ubuntu 16.04 operating system and Python 3.x environment.
For Python library dependencies, please follow the ```requirements.txt```, or run:
```
pip install requirements.txt
```

For pretrained BERT model, we use bert-base-uncased version. The BERT pretrained model is downloaded from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin), the BERT config file is downloaded from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json), the BERT vocab file is downloaded from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt). Then
- Rename ```bert-base-uncased-pytorch_model.bin``` to ```pytorch_model.bin```
- Rename ```bert-base-uncased-config.json``` to ```config.json```
- Rename ```bert-base-uncased-vocab.txt``` to ```vocab.txt```

and place ```model```, ```config``` and ```vocab``` files into the ```pretrain/bert/base-uncased/``` directory.

## Usage
### Training
For clarification prediction task, run:
```
sh train_classifier.sh
```
The trained model will be stored at the ```log/log_classifier/``` directory. Our trained model can be downloaded from [here](https://drive.google.com/file/d/1xETZwsOv8saR16btWJlBupNShIC4K9rC/view?usp=sharing), please place the unpacked files into the ```log/log_classifier/``` directory.

For question ranking task, run:
```
sh train_ranker.sh
```
The trained model will be stored at the ```log/log_ranker/``` directory. Our trained model can be downloaded from [here](https://drive.google.com/file/d/1oOUxenTDBSYBfzGeAHN0HA7SveqFDzUv/view?usp=sharing), please place the unpacked files into the ```log/log_ranker/``` directory.

### Testing
For clarification need predcition, please first set the ```--test_path``` in ```test_classifier.sh``` to specify the file to be testing. The run:
```
sh test_classifier.sh
```
The output run file will be located at ```output/[dev/test]_clari_need.txt```.

For question ranking prediction, please first set the ```--test_path``` in ```test_ranker.sh``` to specify the file to be testing. Then run:
```
sh test_ranker.sh
```
The output run file will be located at ```output/[dev/test]_ranked_q.txt```.

### Evaluation
Below, we give some examples of how to use the official evaluation script to show our model's performace on dev set. 
For evaluation on clarification need , please first set ```--experiment_type``` and ```--run_file``` in ```eval_clarification_need.sh```, then run:
```
sh eval_clarification_need.sh
```
On dev set, it would produce the output below:
```
Precision:  0.54
Recall:  0.54
F1: 0.54
```

For evaluation on document relevance, please first set ```--experiment_type``` and ```--run_file``` in ```eval_document_relevance.sh```, then run:
```
sh eval_document_relevance.sh
```
On dev set, it would produce the output below:
```
NDCG1: 0.20364583333333333
NDCG3: 0.18494246646500168
NDCG5: 0.17623501136078207
NDCG10: 0.16108532915720883
NDCG20: 0.14571769947324514
P1: 0.2625
P3: 0.21666666666666665
P5: 0.20125
P10: 0.169375
P20: 0.1284375
MRR100: 0.349673813551919
```

For evaluation on question relevance, please first set ```--experiment_type``` and ```--run_file``` in ```eval_question_relevance.sh```, then run:
```
sh eval_question_relevance.sh
```
On dev set, it would produce the output below:
```
Recall5: 0.3522886732290757
Recall10: 0.6246779448621554
Recall20: 0.7353664421234081
Recall30: 0.7636166093967951
```