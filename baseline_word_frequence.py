import argparse
import os
import time
import datetime
import json
from collections import defaultdict
from pathlib import Path
from unicodedata import category
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
...
import jieba.analyse
import utils
import operator
import math
from tqdm import tqdm

def feature_select_tf_idf(one_of_all, full_corpus):
    #总词频统计
    doc_frequency=defaultdict(int)
    for word in one_of_all:
        doc_frequency[word]+=1
 
    #计算每个词的TF值
    word_tf={}  #存储没个词的tf值
    for word in doc_frequency:
        word_tf[word]=doc_frequency[word]/sum(doc_frequency.values())
 
    #计算每个词的IDF值
    doc_num=len(full_corpus)
    word_idf={} #存储每个词的idf值
    word_doc=defaultdict(int) #存储包含该词的文档数
    for word in doc_frequency:
        for doc in full_corpus:
            if word in doc:
                word_doc[word]+=1
    for word in doc_frequency:
        word_idf[word]=math.log(doc_num/(word_doc[word]+1))
 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    for word in doc_frequency:
        word_tf_idf[word]=word_tf[word]*word_idf[word]
 
    # 对字典按值由大到小排序
    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select

def training_loop(train_loader, val_loader, test_loader):
    categories = {}
    full_corpus = [dct['cws'] for loader in (train_loader, val_loader, test_loader) for dct in loader]
    if not os.path.exists(config.cache):
        cate_sentences = {}
        for dct in train_loader:
            cws = dct['cws']
            label = dct['label_id']
            cate_sentences[label] = cate_sentences.get(label, []) + cws
        
        # keywords = jieba.analyse.extract_tags(text, topK=15, withWeight=False, allowPOS=())
        for cate, list_words in tqdm(cate_sentences.items(), total=len(cate_sentences), desc="Get Categories' keywords:"):
            keys = feature_select_tf_idf(list_words, full_corpus)
            extra_keys = jieba.analyse.extract_tags(' '.join(list_words), topK=15, withWeight=False, allowPOS=())
            categories[cate] = list(set([i[0] for i in keys[:15]] + extra_keys))

        with open(config.cache, 'w') as file:
            file.write(json.dumps(categories, ensure_ascii=False, indent=4))
    else:
        categories = json.load(open(config.cache,'r'))
    ...
    val_stats, val_prediction = evaluate(val_loader, categories)
    test_stats, test_prediction = evaluate(test_loader, categories)
    utils.write_json(config.output_dir, "val_prediction", val_prediction)
    utils.write_json(config.output_dir, "test_prediction", test_prediction)
    print(val_stats)
    print(test_stats)
    ...



def evaluate(data_loader, categories):
    # test
    pred_class = []
    goldens = []
    for dct in tqdm(data_loader, total=len(data_loader), desc="Evaluate:"):
        cws = dct['cws']
        label = dct.get('label_id', -1)
        goldens.append(label)
        statis = {}
        for cate, keys in categories.items():
            statis[cate] = sum([1 if key in cws else 0 for key in keys])
        pred = sorted(statis.items(), key=operator.itemgetter(1),reverse=True)[0]
        pred_class.append(int(pred[0]))

    accuracy = accuracy_score(pred_class, goldens)
    precision = precision_score(goldens, pred_class, average='macro')
    recall = recall_score(goldens, pred_class, average='macro')
    F1 = f1_score(goldens, pred_class, average='macro')

    eval_result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F-1': F1
    }

    predict_result = []
    for dct, pred in zip(data_loader, pred_class):
        dct['prediction'] = pred
        dct['label_keys'] = " ".join(categories[str(pred)])
        dct['cws'] = " ".join(dct['cws'])
        predict_result.append(dct)

    return eval_result, predict_result


def data_prepare():

    train_loader, val_loader, test_loader = [json.load(open(file, 'r')) for file in (config.train_file, config.val_file, config.test_file)]
    return train_loader, val_loader, test_loader


def main():

    #### Dataset ####
    train_loader, val_loader, test_loader = data_prepare()

    training_loop(train_loader,val_loader, test_loader)


def parse_args():
    # See: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    parser = argparse.ArgumentParser(
        description="necessarily parameters for run this code."
    )
    parser.add_argument('--config', default='configs/baseline_word_frequence.yaml')
    parser.add_argument('--output_dir', default='output/baseline_word_frequence_debug')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--eval_before_train', action='store_true')
    parser.add_argument('--only_dev', action='store_false')
    parser.add_argument('--dist_backend', default='nccl')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='gradient accumulation for increase batch virtually.')
    parser.add_argument('--max_grad_norm', default=5.0, type=float,
                        help='clip gradient norm of an iterable of parameters')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='device number of current process.') 
    parser.add_argument('--logging_step', default=500, type=int) 
    parser.add_argument('--logging_strategy', type=str, choices=['no','epoch','steps'], default='steps')
    parser.add_argument('--logging_level', type=str, choices=['DEBUG','INFO','ERROR','WARNING'], default='DEBUG')
    parser.add_argument('--save_every_checkpoint', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # set configuration for training or evaluating
    args = parse_args()
    config = utils.read_yaml(args.config)
    config = utils.AttrDict(config)
    args = utils.AttrDict(args.__dict__)
    # The parameters passed in from the command line take precedence
    config.update(args)

    main()
