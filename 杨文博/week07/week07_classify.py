import pandas as pd
import jieba
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Any
from tqdm import tqdm
import re


class Config:
    CSV_PATH = Path(r"D:\work\code\practice\home_work\杨文博\week07\DMSC.csv")
    JIEBA_DATA_PATH = Path(r"D:\work\code\practice\home_work\杨文博\week07\data\jieba.pkl")  # 使用.pkl扩展名更明确
    MIN_WORDS = 10
    MAX_WORDS = 150


class CommentClassifier(ABC):
    @staticmethod
    @abstractmethod
    def classify(data: pd.DataFrame) -> List[Tuple[List[str], int]]:
        """
        抽象分类方法
        :param data: 包含评论和评分的DataFrame
        :return: 返回分词结果和对应标签的列表
        """
        pass


class JiebaClassifier:
    @staticmethod
    def clean_text(text: str) -> str:
        """去除标点符号和空格"""
        # 匹配所有非中文字符、非字母、非数字的字符（包括标点和空格）
        return re.sub(r'[^\w\u4e00-\u9fff]+', '', text)

    @staticmethod
    def classify(data: pd.DataFrame) -> List[Tuple[List[str], int]]:
        # 预处理：过滤空值并转换类型
        clean_data = data[["Comment", "Star"]].dropna()
        comments = clean_data["Comment"].astype(str).tolist()
        stars = clean_data["Star"].astype(int).tolist()

        # 批量清洗和分词（使用tqdm显示进度）
        word_lists = []
        for comment in tqdm(comments, desc="清洗和分词进度"):
            # 先清洗文本
            cleaned_text = JiebaClassifier.clean_text(comment)
            # 再分词
            words = jieba.lcut(cleaned_text)
            # 去除分词后可能存在的空字符串
            words = [word for word in words if word.strip()]
            word_lists.append(words)

        # 向量化处理标签
        labels = [0 if star < 4 else 1 for star in stars]

        # 组合并过滤结果
        return [
            (words, label)
            for words, label in zip(word_lists, labels)
            if Config.MIN_WORDS <= len(words) <= Config.MAX_WORDS
        ]


class DataProcessor:
    def __init__(self):
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """加载数据并进行基本清洗"""
        if not Config.CSV_PATH.exists():
            raise FileNotFoundError(f"CSV文件不存在: {Config.CSV_PATH}")

        data = pd.read_csv(Config.CSV_PATH)
        # 基本数据清洗
        data = data.dropna(subset=["Comment", "Star"])
        return data

    def process_and_save(self, classifier: CommentClassifier) -> None:
        """
        处理数据并保存结果
        :param classifier: 评论分类器实例
        """
        processed_data = classifier.classify(self.data)

        # 确保输出目录存在
        Config.JIEBA_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(Config.JIEBA_DATA_PATH, "wb") as f:
            pickle.dump(processed_data, f)

        print(f"数据处理完成，已保存到: {Config.JIEBA_DATA_PATH}")
        print(f"处理后的数据量: {len(processed_data)}")


if __name__ == '__main__':
    try:
        processor = DataProcessor()
        classifier = JiebaClassifier()
        processor.process_and_save(classifier)
    except Exception as e:
        print(f"程序运行出错: {e}")
