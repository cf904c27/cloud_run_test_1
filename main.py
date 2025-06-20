#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import io
import sys
sys.path.append('../')

import json
import random
import requests
import functools
import numpy as np
import pandas as pd
from flask import jsonify, send_file, make_response

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer


import warnings

def preprocess_function(examples, tokenizer, max_seq_length, label_nums, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        examples(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_length(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        label_nums(obj:`int`): The number of the labels.
    Returns:
        result(obj:`dict`): The preprocessed data including input_ids, token_type_ids, labels.
    """
    result = tokenizer(text=examples["sentence"], max_seq_len=max_seq_length)
    # One-Hot label
    if not is_test:
        result["labels"] = [float(1) if i in examples["label"] else float(0) for i in range(label_nums)]
    return result


def read_local_dataset(path, label_list=None, is_test=False):
    """
    Read dataset
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:
                items = line.strip().split("\t")
                sentence = "".join(items)
                yield {"sentence": sentence}
            else:
                items = line.strip().split("\t")
                if len(items) == 0:
                    continue
                elif len(items) == 1:
                    sentence = items[0]
                    labels = []
                else:
                    sentence = "".join(items[:-1])
                    label = items[-1]
                    labels = [label_list[l] for l in label.split(",")]
                yield {"sentence": sentence, "label": labels}





def product_predict(request):
    
    data = request.get_json()

    print(paddle.__version__)

    device = 'cpu'
    max_seq_length = 256
    batch_size = 1
    params_path = 'exp45'

    label_map ={'3C':6,
    '家電':7,
    '其他':99,
    '美妝保養':2,
    '生活':4,
    '服飾配件':3,
    '旅遊':5,
    '毛孩':10,
    '新奇玩樂':5,
    '媽咪寶貝':8,
    '營養保健':9,
    '美食':1,
    '':99}
        
    for k,v in data.items():
        
        with open('/tmp/data.txt', 'w', encoding="utf-8") as f:
            for line in eval(v):
                f.write(f"{line}\n")
                print(f"{line}\n")
            
            # 獲取檔案大小
            file_size = f.tell()

        if file_size > 0:
            warnings.warn(f"Data has been written to data.txt. File size: {file_size} bytes")
        else:
            warnings.warn("Warning: data.txt is empty after writing")

        paddle.set_device(device)
        model = AutoModelForSequenceClassification.from_pretrained(params_path)
        tokenizer = AutoTokenizer.from_pretrained(params_path)

        label_list = []
        label_path = 'label.txt'
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                label_list.append(line.strip())
                print(line.strip())


        ######################################
        data_ds = load_dataset(
            read_local_dataset, path='/tmp/data.txt', is_test=True, lazy=False
        )

        trans_func = functools.partial(
            preprocess_function,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_nums=len(label_list),
            is_test=True,
        )

        data_ds = data_ds.map(trans_func)

        # batchify dataset
        collate_fn = DataCollatorWithPadding(tokenizer)
        data_batch_sampler = BatchSampler(data_ds, batch_size=batch_size, shuffle=False)
        warnings.warn("Warning: batchify dataset")

        data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)
        ######################################

        warnings.warn("Start Predict")
        
        results = []
        model.eval()
        for batch in data_data_loader:
            logits = model(**batch)
            probs = F.sigmoid(logits).numpy()
            for prob in probs:
                labels = []
                if max(prob) > 0.8:
                    labels.append(np.argmax(prob))
                results.append(labels)

        warnings.warn("End of Predict")

        
        text = []
        prediction = []

        for d, result in zip(data_ds.data, results):
            label = [label_list[r] for r in result]
            label_txt = []
            if len(label) == 0:
                label_txt.append('99')
            else:
                for l in label:
                    q = label_map[l]
                    label_txt.append(str(q))

                    warnings.warn(f"Text: {str(q)}")

            prediction.append(','.join(label_txt))
            text.append(d["sentence"])

            
        
        ######################################
        predict_dict = dict()
        

        if len(prediction) > 0:
            data = pd.DataFrame()
            data['text']= text
            updated_predictions= []
            for t in text:
                if str(t).isdigit():
                    updated_predictions.append(99)
                else:
                    updated_predictions.append(prediction[text.index(t)])
            data['prediction'] = updated_predictions
            jdata = data.to_json(orient='records', force_ascii=False)
            predict_dict.update({k:json.loads(jdata)})
        else:
            data = pd.DataFrame()
            data['text']= text
            data['prediction']= 99
            
            jdata = data.to_json(orient='records', force_ascii=False)
            predict_dict.update({k:json.loads(jdata)})

        warnings.warn(f"Prediction completed for key: {k}")
            
        return jsonify(predict_dict)
