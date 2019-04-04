"""
author :wangshengguang
date:2019-03-27
train wide & deep model
"""
import tensorflow as tf

from data_helper import input_fn, read_data


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
    gain_bucket = tf.feature_column.bucketized_column(captial_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    loss_bucket = tf.feature_column.bucketized_column(captial_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    # 交叉特征
    cross_columns = [
        tf.feature_column.crossed_column([age_bucket, gain_bucket], hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket, loss_bucket], hash_bucket_size=16),
    ]

    # 特征
    base_columns = [gender, work_class, education, marital_status, occupation, relationship, age_bucket, gain_bucket,
                    loss_bucket]
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


def build_model_estimator(wide_column, deep_column, model_folder):
    """
    :param wide_column: wide侧特征
    :param deep_column: deep侧特征
    :param model_folder: 模型输出文件夹
    :return:
            model_es,serving_input_fn
    """
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,
        linear_feature_columns=wide_column,
        linear_optimizer=tf.train.FtrlOptimizer(0.1, l2_regularization_strength=1.0),
        dnn_feature_columns=deep_column,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001),
        dnn_hidden_units=[128, 64, 32, 16],  # 隐层层数，维度；深度学习：VC维=参数数目=样本数；机器学习：VC维=参数数目=10*样本数
    )
    feature_column = wide_column + deep_column
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return model_es, serving_input_fn


def run_main(train_file, test_file, model_floder, model_export_floder):
    wide_column, deep_column = get_feature_column()
    model, serving_input_fn = build_model_estimator(wide_column, deep_column, model_export_floder)
    model.train(input_fn=input_fn(train_file, 20, True, 100))
    # model.evaluate(input_fn=input_fn(test_file, 1, False, 100))
    # model.export_savedmodel(model_export_floder, serving_input_fn)
    tf_data, test_labels = read_data(test_file)
    result = model.predict(input_fn=input_fn(test_file, num_epochs=1, shuffle=False, batch_size=1))
    result = [a for a in result]
    predict_list = [one["probabilities"][1] for one in result]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_labels, predict_list)
    print(auc)
    # import ipdb
    # ipdb.set_trace()


if __name__ == "__main__":
    from data_helper import data_dir
    import os

    train_file = os.path.join(data_dir, "adult.data")
    test_file = os.path.join(data_dir, "adult.test")

    run_main(train_file, test_file, "data/wd", "data/wd_export")
