import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_numeric():
    price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x + 2

    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_transformed_tensor = price_column._get_dense_tensor(builder)
    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

    # 使用input_layer
    price_transformed_tensor = feature_column.input_layer(price, [price_column])
    with tf.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor]))


test_numeric()
