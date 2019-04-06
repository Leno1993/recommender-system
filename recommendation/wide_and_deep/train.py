"""
author :wangshengguang
date:2019-03-27
train wide & deep model
"""

import os

import tensorflow as tf

from data_helper import data_dir, get_feature_column, input_fn

# 训练信息输出到屏幕  DEBUG, INFO, WARN, ERROR, or FATAL
# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)  # DEBUG, INFO, WARN, ERROR, or FATAL

# TF_CPP_MIN_LOG_LEVEL = 1  # 默认设置，为显示所有信息
# TF_CPP_MIN_LOG_LEVEL = 2  # 只显示error和warining信息
# TF_CPP_MIN_LOG_LEVEL = 3  # 只显示error信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

train_file = os.path.join(data_dir, "adult.data")
test_file = os.path.join(data_dir, "adult.test")


def build_model_estimator(wide_column, deep_column, model_folder, model_type="wide_deep"):
    """
    :param wide_column: wide侧特征
    :param deep_column: deep侧特征
    :param model_folder: 模型输出文件夹
    :return:
            model_es,serving_input_fn
    """
    hidden_units = [128, 64, 32, 16]  # 128*64*32*16=4194304
    if model_type == "wide":
        model = tf.estimator.LinearClassifier(
            model_dir=model_folder,
            feature_columns=wide_column,
            optimizer=tf.train.FtrlOptimizer(0.1, l2_regularization_strength=1.0)
        )
    elif model_type == "deep":
        model = tf.estimator.DNNClassifier(
            model_dir=model_folder,
            feature_columns=wide_column,
            hidden_units=hidden_units,
            optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                        l2_regularization_strength=0.001)
        )
    else:
        model = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_folder,
            linear_feature_columns=wide_column,
            linear_optimizer=tf.train.FtrlOptimizer(0.1, l2_regularization_strength=1.0),
            dnn_feature_columns=deep_column,
            dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                            l2_regularization_strength=0.001),
            dnn_hidden_units=hidden_units,  # 隐层层数，维度；深度学习：VC维=参数数目=样本数；机器学习：VC维=参数数目=10*样本数
        )
    feature_column = wide_column + deep_column
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    return model, serving_input_fn


def train(wide_column, deep_column, model_floder, model_type="wide_deep"):
    model, serving_input_fn = build_model_estimator(wide_column, deep_column, model_floder, model_type="wide_deep")
    # model.train(input_fn=input_fn(train_file, num_epochs=20, shuffle=True, batch_size=100), steps=None)
    evaluate_result = model.evaluate(input_fn=input_fn(test_file, num_epochs=1, shuffle=False, batch_size=100))
    print("*** model_type: {}, accuracy： {:.4f}， auc: {:.4f}".format(
        model_type, evaluate_result["accuracy"], evaluate_result["auc"]))
    # model.export_savedmodel(model_export_floder, serving_input_fn)
    # tf_data, test_labels = read_data(test_file)
    # result = model.predict(input_fn=input_fn(test_file, num_epochs=1, shuffle=False, batch_size=1))
    # result = [a for a in result]
    # predict_list = [one["probabilities"][1] for one in result]
    # from sklearn.metrics import roc_auc_score
    # auc = roc_auc_score(test_labels, predict_list)
    # print("** roc_auc_score: {}".format(auc))
    print("\n")


def main():
    wide_column, deep_column = get_feature_column()
    model_types = ["wide", "deep", "wide_deep"]
    model_dirs = [os.path.join(data_dir, model_type) for model_type in model_types]
    for model_dir, model_type in zip(model_dirs, model_types):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        train(wide_column, deep_column, model_dir, model_type)
    # import ipdb
    # ipdb.set_trace()


if __name__ == "__main__":
    main()
