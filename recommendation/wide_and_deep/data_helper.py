import os

import pandas as pd
import requests
import tensorflow as tf

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(base_dir, "data")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def download_data():
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/{}'
    train_file = "adult.data"
    test_file = "adult.test"
    train_data_url = base_url.format(train_file)
    test_data_url = base_url.format(test_file)
    for name, url in [(train_file, train_data_url), (test_file, test_data_url)]:
        file_path = os.path.join(data_dir, name)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                content = requests.get(url).text
                f.write(content)


_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]  # 原始数据CSV文件，列名

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]  # CSV文件每列默认值


def read_data(data_file, test=False):
    tf_data = pd.read_csv(
        tf.gfile.Open(data_file),
        names=_CSV_COLUMNS,
        skip_blank_lines=True, skipinitialspace=True,
        engine="python",
        skiprows=1
    )
    tf_data = tf_data.dropna(how="any", axis=1)  # remove Na elements
    labels = tf_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
    return tf_data, labels


def input_fn(data_file, num_epochs, shuffle, batch_size):
    tf_data, labels = read_data(data_file)
    return tf.estimator.inputs.pandas_input_fn(
        x=tf_data,
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1
    )
