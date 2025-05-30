#! -*- coding:utf-8 -*-
# 情感分类任务, 加载bert权重
# Mixup策略，包含embedding，hidden, encoder的mixup

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.layers import MixUp
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

maxlen = 256
batch_size = 16
config_path = 'E:/data/pretrain_ckpt/google-bert/bert-base-chinese/bert4torch_config.json'
checkpoint_path = 'E:/data/pretrain_ckpt/google-bert/bert-base-chinese/pytorch_model.bin'
dict_path = 'E:/data/pretrain_ckpt/google-bert/bert-base-chinese/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'train'  # train表示训练，infer表示推理
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
    batch_token_ids, batch_labels = [], []
    for text, label in batch:
        token_ids = tokenizer.encode(text, maxlen=maxlen)[0]
        batch_token_ids.append(token_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return batch_token_ids, batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(['F:/data/corpus/sentence_classification/sentiment/sentiment.train.data']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(['F:/data/corpus/sentence_classification/sentiment/sentiment.valid.data']), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(['F:/data/corpus/sentence_classification/sentiment/sentiment.test.data']),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, mixup_method='encoder', pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, segment_vocab_size=0)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)
        self.mixup = MixUp(method=mixup_method)

    def forward(self, token_ids):
        hidden_states, pooling = self.mixup.encode(self.bert, [token_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        y_pred = self.dense(output)
        return y_pred
    
    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_states, pooling = self.bert([token_ids])
            pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
            output = self.dropout(pooled_output)
            y_pred = self.dense(output)
        return y_pred

model = Model().to(device)

class Loss(nn.Module):
    def forward(self, y_pred, y_true):
        return model.mixup(nn.CrossEntropyLoss(), y_pred, y_true)
    
# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=Loss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total


if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
    else:
        model.load_weights('best_model.pt')
