# 搜狐基于实体的情感分类
- 比赛链接：https://www.biendata.xyz/competition/sohu_2022/

| 解决方案 | 链接 | 指标 |
| ---- | ---- | ---- |
| Top1 | [知乎](https://zhuanlan.zhihu.com/p/533808475)| 初赛f1=0.7253, 复赛f1=0.8173 |
| baseline | —— | 初赛f1=0.6737 |

# bert4torch复现
- 预训练模型使用xlnet
- 由于比赛结束无法提交，复现只使用线下dev作为对比
- dev为前2000，未使用方案中的后10%作为dev, dev指标略微有点不稳定

| 复现方案 | 方案 | 指标 |
| ---- | ---- | ---- |
| Top1_github | 前2000为dev, 不使用swa, 有warmup, 无label_smoothing, 无fgm, 梯度累积=3, 无rdrop | Epoch 4/10: f1=0.7697|
| Top1_bert4torch复现1 | 参数同上 | Epoch 10/10: f1=0.7556 |
| Top1_bert4torch复现2 | 参数同上+fgm+swa | Epoch 5/10: f1=0.7877 |


| Top1_github | Top1_bert4torch复现1 | Top1_bert4torch复现2 |
| ---- | ---- | ---- |
| 0.728  | 0.7039 | 0.0274 |
| 0.7198 | 0.7327 | 0.7180 |
| 0.747	 | 0.7531 | 0.7453 |
| 0.7625 | 0.7466 | 0.7594 |
| **0.7697** | 0.7464 | **0.7877** |
| 0.7638 | 0.7272 | 0.7726 |
| 0.7415 | 0.7471 | 0.7804 |
| 0.7593 | **0.7556** | 0.7829 |
| 0.7477 | 0.7455 | 0.7697 |
| 0.7466 | 0.7471 | 0.7620 |