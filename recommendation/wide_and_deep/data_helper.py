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


def get_feature_column():
    """
    连续值放入deep侧，离散值放入wide侧；
    具体处理：
    连续值离散化后放入wide侧
    离散值：
    hash，然后embeeding，放入wide侧
    :return:
    """
    # 连续值处理
    age = tf.feature_column.numeric_column("age")
    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        "gender", ["Female", "Male"])
    education_num = tf.feature_column.numeric_column("education_num")
    captial_gain = tf.feature_column.numeric_column("capital_gain")
    captial_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")
    # 离散值处理
    work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=512)
    relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=512)

    # 连续值离散化
    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60])
    captial_gain_bucket = tf.feature_column.bucketized_column(captial_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    captial_loss_bucket = tf.feature_column.bucketized_column(captial_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    # 交叉特征
    cross_columns = [
        tf.feature_column.crossed_column([age_bucket, captial_gain_bucket], hash_bucket_size=36),
        tf.feature_column.crossed_column([captial_gain_bucket, captial_loss_bucket], hash_bucket_size=16),
    ]

    # 特征
    base_columns = [gender, work_class, education, marital_status, occupation, relationship, age_bucket,
                    captial_gain_bucket, captial_loss_bucket]
    wide_columns = base_columns + cross_columns
    deep_columns = [
        # 连续值
        age,
        education_num,
        captial_gain,
        captial_loss,
        hours_per_week,
        # 离散值的 embedding
        tf.feature_column.embedding_column(work_class, 9),
        tf.feature_column.embedding_column(education, 9),
        tf.feature_column.embedding_column(marital_status, 9),
        tf.feature_column.embedding_column(occupation, 9),
        tf.feature_column.embedding_column(relationship, 9)
    ]
    return wide_columns, deep_columns
