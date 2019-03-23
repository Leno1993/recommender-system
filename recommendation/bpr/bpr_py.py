# Implement BPR.
# Steffen Rendle, et al. BPR: Bayesian personalized ranking from implicit feedback.
# Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI, 2009.
# @author Runlong Yu, Mingyue Cheng, Weibo Gao
import glob
import os
import re

import numpy as np

from data_helper import DataHelper, data_dir, largest_indices

checkpoint_dir = os.path.join(data_dir, "checkpoint_py")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


class BPR(object):
    def __init__(self):
        self.data_helper = DataHelper()

    def load_model(self, saved_model_path=""):
        """模型加载"""
        if not saved_model_path:
            saved_model_path = self.get_min_loss_model_path()
        U, V = np.load(saved_model_path)
        return U, V

    def get_min_loss_model_path(self, checkpoint_dir=checkpoint_dir):
        # 获取loss最小的模型的路径
        min_loss_model_paths = sorted(
            glob.glob(os.path.join(checkpoint_dir, "*.npy")),
            key=lambda model_path: float(re.findall(r"(\d{4})\.npy", model_path)[0]))
        return min_loss_model_paths[0]

    def train_step(self, batch_samples, update_parm=True):
        """一次训练迭代过程"""
        u = batch_samples[:, 0] - 1  # 编号-1成索引
        i = batch_samples[:, 1] - 1
        j = batch_samples[:, 2] - 1
        r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
        r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
        r_uij = r_ui - r_uj
        loss_func = -1.0 / (1 + np.exp(r_uij))
        loss_func = np.mean(loss_func)
        # update U and V
        if update_parm:
            self.U[u] += -self.lr * (loss_func * (self.V[i] - self.V[j]) + self.reg * self.U[u])
            self.V[i] += -self.lr * (loss_func * self.U[u] + self.reg * self.V[i])
            self.V[j] += -self.lr * (loss_func * (-self.U[u]) + self.reg * self.V[j])
            # update biasV
            self.biasV[i] += -self.lr * (loss_func + self.reg * self.biasV[i])
            self.biasV[j] += -self.lr * (-loss_func + self.reg * self.biasV[j])
        return loss_func

    def train(self):
        """训练"""
        self.latent_factors = 20
        self.lr = 0.01
        self.reg = 0.01  # 正则化参数
        max_u_id = max(self.data_helper.user_ratings.keys())
        max_i_id = max(self.data_helper.item_ids)
        self.U = np.random.rand(max_u_id, self.latent_factors) * 0.01
        self.V = np.random.rand(max_i_id, self.latent_factors) * 0.01
        self.biasV = np.random.rand(max_i_id) * 0.01
        step = 0
        for batch_samples in self.data_helper.generate_samples(epoch_num=10, batch_size=3):
            train_loss = self.train_step(batch_samples)
            step += 1
            if step % 10000 == 0:
                loss = []
                for batch_samples in self.data_helper.generate_samples(batch_size=1000, data_type="validate"):
                    _loss = self.train_step(batch_samples, update_parm=False)
                    loss.append(_loss)
                val_loss = np.mean(loss)
                print("step:{}, loss: {}, val_loss:{}".format(step, train_loss, val_loss))
                model_path = os.path.join(checkpoint_dir, "{}_{:.4f}.npy".format(step, val_loss))
                np.save(model_path, [self.U, self.V])

    def predict(self, user_id):
        """预测推荐"""
        U, V = self.load_model()
        _predict = np.dot(U[user_id], V.T)
        _predict = largest_indices(_predict, 3)
        return _predict


if __name__ == '__main__':
    bpr = BPR()
    bpr.train()
    pred = bpr.predict(0)
    print("predict:{}".format(pred))
