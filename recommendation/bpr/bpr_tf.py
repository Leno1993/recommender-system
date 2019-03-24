import os

import numpy as np
import tensorflow as tf

from data_helper import DataHelper, data_dir, largest_indices

checkpoint_dir = os.path.join(data_dir, "checkpoint_tf")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def bpr_mf(user_count, item_count, hidden_dim):
    """模型定义"""
    u = tf.placeholder(tf.int32, [None], name="input_u")
    i = tf.placeholder(tf.int32, [None], name="input_i")
    j = tf.placeholder(tf.int32, [None], name="input_j")
    # with tf.device("/cpu:0"):
    user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    # MF predict: u_i > u_j
    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)
    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])
    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
    return u, i, j, mf_auc, bprloss, train_op


with tf.Graph().as_default(), tf.Session() as session:
    data_helper = DataHelper()
    user_count = max(data_helper.user_ratings.keys())
    item_count = max(data_helper.item_ids)
    hidden_dim = 20
    epoch_num = 10
    batch_size = 16

    u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, hidden_dim)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # 初始化参数
    session.run(tf.global_variables_initializer())


    def train():
        """训练"""
        step = 0
        for uij in data_helper.generate_samples(epoch_num=epoch_num, batch_size=batch_size, data_type="train"):
            loss, _train_op = session.run([bprloss, train_op],
                                          feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
            step += 1
            if step % 1000 == 0:
                print("step:{}, loss: {:.4f}".format(step, loss))
                path = saver.save(session, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))


    def test():
        """测试"""
        auc, loss = [], []
        for t_uij in data_helper.generate_samples(data_type="test", batch_size=1000):
            _auc, _test_bprloss = session.run([mf_auc, bprloss],
                                              feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]})
            auc.append(_auc), loss.append(_test_bprloss)
        print("test loss:{:.4f}, test auc:{:.4f} ".format(np.mean(auc), np.mean(loss)))
        print("")
        return loss, auc


    # 执行训练和测试
    train()
    test()


def load_model():
    """模型加载"""
    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        user_emb_w, item_emb_w = sess.run(
            [graph.get_tensor_by_name('user_emb_w:0'), graph.get_tensor_by_name("item_emb_w:0")])
    return user_emb_w, item_emb_w


def predict(user_id):
    """预测推荐"""
    U, V = load_model()
    _predict = np.dot(U[user_id - 1], V.T)  # index = user_id-1
    _predict = largest_indices(_predict, 3) + 1  # top 3 ; user_id = index+1
    return _predict


user_id = 1
pred = predict(user_id)

print("user_id:{}, predict: {}".format(user_id, pred))
