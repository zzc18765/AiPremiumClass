import fasttext
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
import io
import jieba
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#本文档读取西游记数据
'''
with open('xi_you.txt','r',encoding='utf-8') as f:
    context=f.read()
with open('xi_you_parse.txt','w',encoding='utf-8') as ff:
    #以空格作为分界并且分词，构造输入文本
    ff.write(' '.join(jieba.lcut(context)))
print(context)
'''
#无监督学习
model = fasttext.train_unsupervised('xi_you_parse.txt')
# 计算词汇间的相关度
word1 = '悟空'
word2 = '行者'
word3='齐天大圣'
#评估两个词的相关度
print(model.get_word_vector(word1))
print(model.get_word_vector(word2))
print(model.get_word_vector(word3))
#评估相关度
similarity = model.get_word_vector(word1).dot(model.get_word_vector(word2)) / (
    np.linalg.norm(model.get_word_vector(word1)) * np.linalg.norm(model.get_word_vector(word2)))
print(f"{word1} 和 {word2} 的相关度: {similarity}")
# print(model.words)
#找最近的5个邻居
print(model.get_nearest_neighbors('自然语言处理',k=5))
print(model.get_analogies('悟空','行者','齐天大圣'))

'''
'# 选做:使用tensorboard绘制词向量可视化图
log_dir = 'logs/embeddings'
writer = SummaryWriter(log_dir)

# 提取词汇和对应的词向量
words = model.words
vectors = np.array([model.get_word_vector(word) for word in words])

# 使用TSNE进行降维
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# 转换为 PyTorch 张量
vectors_2d_tensor = torch.tensor(vectors_2d, dtype=torch.float32)

# 将词向量和对应的标签写入 TensorBoard
writer.add_embedding(vectors_2d_tensor, metadata=words)

writer.close()

print("词向量可视化数据已保存到", log_dir)
print("请在终端中运行以下命令启动TensorBoard:")
print("tensorboard --logdir=logs/embeddings")'
'''
