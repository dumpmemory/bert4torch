#! -*- coding:utf-8 -*-
# 语义相似度任务-无监督：训练集为网上pretrain数据, dev集为sts-b
# loss: 其实就是SimCSE，只是用了两个模型而已 

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import copy
import random
random.seed(2022)

maxlen = 256
batch_size = 8
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    texts_list = [[] for _ in range(2)]
    for text in batch:
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        texts_list[0].append(token_ids)
        texts_list[1].append(token_ids)

    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.arange(texts_list[0].size(0), device=texts_list[0].device)
    return texts_list, labels

# 加载数据集
def get_data(filename):
    train_data = []
    with open(filename, encoding='utf-8') as f:
        for row, l in enumerate(f):
            if row == 0:  # 跳过首行
                continue
            text = l.strip().replace(' ', '')
            train_data.append(text)
    return train_data

train_data = get_data('F:/Projects/data/corpus/pretrain/film/film.txt')
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
from task_sentence_embedding_stsb_CosineSimilarityLoss import valid_dataloader

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean', scale=20.0):
        super().__init__()
        self.model1, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.model2 = copy.deepcopy(self.model1)
        self.pool_method = pool_method
        self.scale = scale

    def forward(self, token_ids_list):
        token_ids = token_ids_list[0]
        hidden_state1, pooler1 = self.model1([token_ids])
        embeddings_a = get_pool_emb(hidden_state1, pooler1, token_ids.gt(0).long(), self.pool_method)

        token_ids = token_ids_list[1]
        hidden_state2, pooler2 = self.model2([token_ids])
        embeddings_b = get_pool_emb(hidden_state2, pooler2, token_ids.gt(0).long(), self.pool_method)

        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.model1([token_ids])
            output = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
        return output
        
    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
)

# 定义评价函数
def evaluate(data):
    embeddings1, embeddings2, labels = [], [], []
    for (batch_token1_ids, batch_token2_ids), label in data:
        embeddings1.append(model.encode(batch_token1_ids))
        embeddings2.append(model.encode(batch_token2_ids))
        labels.append(label)

    embeddings1 = torch.cat(embeddings1).cpu().numpy()
    embeddings2 = torch.cat(embeddings2).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    eval_pearson_cosine, _ = spearmanr(labels, cosine_scores)
    return eval_pearson_cosine


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = evaluate(valid_dataloader)
        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # model.save_weights('best_model.pt')
        print(f'val_consine: {val_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=20, 
            steps_per_epoch=500, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')