#! -*- coding:utf-8 -*-
# DDP示例
# 启动命令：python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 task_distributed_data_parallel.py

import os
# 也可命令行传入
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModelDDP
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything
import torch.nn as nn
import torch
import torch.optim as optim
import random, os, numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

# 模型设置
maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids, batch_labels.flatten()], None

# 加载数据集
train_dataset = MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.train.data'])
train_sampler = DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构，这里loss并不是放在模型里计算的
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, return_model_config=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.config['hidden_size'], 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, token_ids, segment_ids, labels):
        _, pooled_output = self.bert([token_ids, segment_ids])
        output = self.dropout(pooled_output)
        output = self.dense(output)
        loss = self.loss_fn(output, labels)
        return loss
model = Model().to(device)

# 指定DDP模型使用多gpu, master_rank为指定用于打印训练过程的local_rank
model = BaseModelDDP(model, master_rank=0, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=lambda x, _: x,  # 直接把forward计算的loss传出来
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
)

if __name__ == '__main__':
    model.fit(train_dataloader, epochs=20, steps_per_epoch=None)
