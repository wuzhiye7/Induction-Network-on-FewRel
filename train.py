# -*- coding: utf-8 -*-
"""
Created on: 2019/5/31 14:08
@Author: zsfeng
"""
import sys
from util.data_loader import JSONFileDataLoader
from model.graph import InductionGraph

model_name = 'induction'
N = 5
K = 5
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)

if model_name == 'induction':
    model = InductionGraph(N, K, 5,
                           pred_embed=train_data_loader.word_vec_mat,
                           sequence_length=max_length)
    model.train((train_data_loader, val_data_loader), "checkpoints/inductionNetwork_test")
