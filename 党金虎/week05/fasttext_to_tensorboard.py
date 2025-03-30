"""
完整FastText词向量训练与可视化流程
"""

import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
#import tensorflow as tf
from tensorboard.plugins import projector
import os
import shutil # 用于删除文件


BASE_DIR = "党金虎/week05/fasttext_test_data"
os.makedirs(BASE_DIR, exist_ok=True)

# 方法1：数据准备
def prepare_corpus():
    input_file = os.path.join(BASE_DIR, "doubanbook_top250_comments_fixed.txt")
    output_file = os.path.join(BASE_DIR, "corpus.txt") #  语料库文件

    # 如果存在语料库文件则直接返回
    if os.path.exists(output_file):
        return output_file

    # 原始数据预处理,一行一条评论
    with open(input_file,'r',encoding='utf-8') as f,\
         open(output_file,'w',encoding='utf-8') as w:
        for line in f:
            terms = line.split('\t')
            if len(terms) == 6:
                w.write(terms[5])
    
    print("语料库文件已生成:", output_file)
    return output_file


# 方法2：训练fasttext模型
def train_fasttext(corpus_path):
    # 模型文件
    model_file = os.path.join(BASE_DIR, "fasttext_model.bin")
    if os.path.exists(model_file):
        # 加载已有模型
        return fasttext.load_model(model_file)
    print("训练fasttext中...")
    try:
        model_result = fasttext.train_unsupervised( # 
            input=corpus_path,
            wordNgrams=3)
        model_result.save_model(model_file)
    except TypeError as e:
        print("训练失败:", e)
    print(f"模型已保存到{model_file}")
    return model_result

# 方法3：词汇相似度计算  
def calculate_similarity(model):
    # 词对相似度    
    word_pairs = [("小说","文学"),
                  ("爱情","浪漫"),
                  ("历史","战争"),
                  ("科幻","未来"),
                  ("科技","人工智能")
                  ]
    print("词对相似度:")
    for word1,word2 in word_pairs:
        try:
            vec1 = model.get_word_vector(word1).reshape(1,-1) # 词向量
            vec2 = model.get_word_vector(word2).reshape(1,-1) # 1行n列
            cosine_similarity = cosine_similarity(vec1,vec2) # 计算余弦相似度
            print(f"{word1} - {word2}: {cosine_similarity[0][0]:.4f}")
        except:
            print(f"{word1} - {word2}: 无法计算")
    # 查找最近邻词汇
    target_words = ["小说","历史","爱情","科技"]
    print("最近邻词汇:")
    for word in target_words:
        try:
            print(f"与{word} 最相近的5个词:")
            neighbors = model.get_nearest_neighbors(word,k=5)
            print(f"{word}: {neighbors}")
        except:
            print(f"{word}: 无法计算")

    

if __name__  == '__main__':

    # 1、数据准备
    corpus_path = prepare_corpus()

    # 2、训练fasttext模型
    model = train_fasttext(corpus_path)

    # 3、词汇相似度计算
    calculate_similarity(model)

