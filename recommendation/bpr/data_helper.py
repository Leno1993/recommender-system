import io
import os
import random
import tarfile
import zipfile
from collections import defaultdict

import numpy as np
import requests

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def largest_indices(ary, k):
    """
    :param ary: flatten numpy ndarray
    :param k:  top k largest
    :return: the ordered k largest indices from a numpy array. O(n + k*log k)
    """
    indices = np.argpartition(ary, -k)[-k:]
    indices = indices[np.argsort(-ary[indices])]
    return indices


def maybe_download_and_extract(url, dest_directory=data_dir):
    """下载数据"""
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    # 文件下载
    if not os.path.exists(filepath):
        try:
            r = requests.get(url, stream=True)
            zip_file = zipfile.ZipFile(io.BytesIO(r.content))
            zip_file.extractall(dest_directory)
            # check=zipfile.is_zipfile(io.BytesIO(r.content))
        except:
            raise BaseException("下载失败，请手动下载文件至目录：{}".format(dest_directory))
    else:
        print("文件已存在： {}".format(filepath))
    uncompress(filepath, dest_directory)


def uncompress(filepath, dest_directory='.'):
    if tarfile.is_tarfile(filepath):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    elif zipfile.is_zipfile(filepath):
        f = zipfile.ZipFile(filepath)
        f.extractall(dest_directory)  # 将所有文件解压到channel1目录下


class DataHelper(object):
    def __init__(self, validation_scale=0.1, test_scale=0.1):
        self.train_data, self.validate_data, self.test_data = self.load_data(validation_scale, test_scale)
        self.user_ratings, self.item_ids = self.get_ratings_items()

    def load_data(self, validation_scale=0.1, test_scale=0.1, force_update=True):
        """下载数据，切分数据集"""
        url = "http://link.zhihu.com/?target=http%3A//files.grouplens.org/datasets/movielens/ml-100k.zip"
        maybe_download_and_extract(url)
        # user_id, item_id, rating, timestamp = line.split("\t")
        if force_update or not os.path.exists(os.path.join(data_dir, "train_data.txt")):
            # 重新生成划分数据集
            data_path = os.path.join(data_dir, r'ml-100k', 'u.data')
            with open(data_path, 'r') as f:
                samples = [line.split("\t") for line in f.readlines()]
                # samples = [[int(user_id), int(item_id), int(rating), int(timestamp)]
                #            for user_id, item_id, rating, timestamp in samples]
            self.split_data(samples, validation_scale, test_scale)

        def read_samples(data_type):
            """
            :param data_type:  train,validate,test
            """
            with open(os.path.join(data_dir, "{}_data.txt".format(data_type)), "r") as f:
                data = [line.split(",") for line in f.readlines()]
                samples = [[int(user_id), int(item_id), int(rating), int(timestamp)]
                           for user_id, item_id, rating, timestamp in data]
            return samples

        train_data = read_samples("train")
        validate_data = read_samples("validate")
        test_data = read_samples("test")
        return train_data, validate_data, test_data

    def split_data(self, samples, validation_scale, test_scale):
        """切分数据集"""
        test_num = int(len(samples) * test_scale) if isinstance(
            test_scale, float) else test_scale  # 测试数据所占比重 或者测试数据数目
        validate_num = int(len(samples) * validation_scale) if isinstance(
            validation_scale, float) else validation_scale  # 测试数据所占比重 或者测试数据数目
        random.shuffle(samples)
        train_data = samples[:len(samples) - validate_num - test_num]
        validate_data = samples[len(samples) - validate_num - test_num:len(samples) - validate_num]
        test_data = samples[len(samples) - validate_num:]
        for datas, data_type in [(train_data, "train"), (validate_data, "validate"), (test_data, "test")]:
            with open(os.path.join(data_dir, "{}_data.txt".format(data_type)), "w") as f:
                _lines = [",".join(one_sample) for one_sample in datas]
                f.write("".join(_lines))
                f.flush()

    def get_ratings_items(self):
        """
        :param data_type:  train,validate,test
        :return:
        """
        user_ratings = defaultdict(set)
        data = self.test_data + self.train_data + self.validate_data
        for user_id, item_id, rating, timestamp in data:
            user_ratings[user_id].add(item_id)
        max_u_id = max(user_ratings.keys())
        # max_i_id = max(user_ratings.values())
        user_count = len(user_ratings)
        item_ids = set()
        for item_set in user_ratings.values():
            item_ids |= item_set
        item_count = len(item_ids)
        max_i_id = max(item_ids)
        print("user_ratings: user_count:{}, item_count:{}, max_u_id:{}, max_i_id:{}".format(
            user_count, item_count, max_u_id, max_i_id))
        return user_ratings, list(item_ids)

    def generate_samples(self, batch_size=1, epoch_num=1, shuffle=True, data_type="train"):
        """
        :param type: train test 训练或测试
        :param test_scale: count 或 比例
        :return:
        """
        data = {"train": self.train_data, "validate": self.validate_data, "test": self.test_data}[data_type]
        for epoch_num in range(epoch_num):  # 每个周期数据分多少批数据
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(len(data)))
                shuffled_data = np.array(data)[shuffle_indices]
            else:
                shuffled_data = data
            batch_data = []
            for user_id, item_id, rating, timestamp in shuffled_data:
                unrating_item_id = random.choice(self.item_ids)
                while unrating_item_id in self.user_ratings[user_id]:
                    unrating_item_id = random.choice(self.item_ids)
                batch_data.append([user_id, item_id, unrating_item_id])
                if len(batch_data) == batch_size:
                    yield np.array(batch_data)
                    batch_data = []
