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
        U, V, biasV = np.load(saved_model_path)
        return U, V, biasV

    def get_min_loss_model_path(self, checkpoint_dir=checkpoint_dir):
        # 获取loss最小的模型的路径
        min_loss_model_paths = sorted(
            glob.glob(os.path.join(checkpoint_dir, "*.npy")),
            key=lambda model_path: float(re.findall(r"(\d{4})\.npy", model_path)[0]))
        return min_loss_model_paths[0]

    def train_step(self, batch_samples, update_parm=True):
        """一次训练迭代过程"""
        u = batch_samples[:, 0] - 1  # index = user_id-1
        i = batch_samples[:, 1] - 1
        j = batch_samples[:, 2] - 1
        r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]  # predict prob
        r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]  # predict prob
        r_uij = r_ui - r_uj
        loss_func = 1.0 / (1 + np.exp(r_uij))
        loss_func = np.mean(loss_func)
        # update U and V
        if update_parm:
            self.U[u] += self.learning_rate * (loss_func * (self.V[i] - self.V[j]) - self.reg * self.U[u])
            self.V[i] += self.learning_rate * (loss_func * self.U[u] - self.reg * self.V[i])
            self.V[j] += self.learning_rate * (-loss_func * self.U[u] - self.reg * self.V[j])
            # update biasV
            self.biasV[i] += self.learning_rate * (loss_func - self.reg * self.biasV[i])
            self.biasV[j] += self.learning_rate * (-loss_func - self.reg * self.biasV[j])
        return loss_func

    def train(self):
        """训练"""
        self.latent_factors = 20
        self.learning_rate = 0.01
        self.reg = 0.01  # 正则化参数
        max_u_id = max(self.data_helper.user_ratings.keys())
        max_i_id = max(self.data_helper.item_ids)
        self.U = np.random.rand(max_u_id, self.latent_factors) * 0.01
        self.V = np.random.rand(max_i_id, self.latent_factors) * 0.01
        self.biasV = np.random.rand(max_i_id) * 0.01
        step = 0
        for batch_samples in self.data_helper.generate_samples(epoch_num=10, batch_size=8):
            train_loss = self.train_step(batch_samples)
            step += 1
            if step % 10000 == 0:
                # validate, 使用验证集数据选择模型
                loss = []
                for batch_samples in self.data_helper.generate_samples(batch_size=1000, data_type="validate"):
                    _loss = self.train_step(batch_samples, update_parm=False)
                    loss.append(_loss)
                val_loss = np.mean(loss)
                print("step:{}, loss: {:.4f}, val_loss:{:.4f}".format(step, train_loss, val_loss))
                model_path = os.path.join(checkpoint_dir, "{}_{:.4f}.npy".format(step, val_loss))
                np.save(model_path, [self.U, self.V, self.biasV])
        print("\n")

    def test(self):
        def samples_auc(batch_samples):
            u = batch_samples[:, 0] - 1  # index = user_id-1
            i = batch_samples[:, 1] - 1
            j = batch_samples[:, 2] - 1
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]  # predict
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]  # predict
            r_uij = r_ui - r_uj
            return r_uij

        auc = []
        loss = []
        for batch_samples in self.data_helper.generate_samples(batch_size=1000, data_type="test"):
            r_uij = samples_auc(batch_samples)
            _loss = self.train_step(batch_samples, update_parm=False)
            auc.append(r_uij)
            loss.append(_loss)
        auc = np.array(auc)
        test_auc = np.sum(auc > 0) / np.prod(auc.shape)
        test_loss = np.mean(loss)

        print("test_loss:{:.4f}, test_auc:{:.4f}\n".format(test_loss, test_auc))

    def predict(self, user_id, topk=3):
        """预测推荐"""
        U, V, biasV = self.load_model()
        _predict = np.dot(U[user_id - 1], V.T) + biasV  # index = user_id-1
        top_n_index = largest_indices(_predict, topk)  # top 3
        # score = _predict[top_n_index]
        # pred = list(zip(top_n_index + 1, score))  # user_id = index+1
        pred = top_n_index + 1  # user_id = index+1
        return pred


if __name__ == '__main__':
    bpr = BPR()
    bpr.train()
    bpr.test()
    user_id = 10
    pred = bpr.predict(user_id)
    print("user_id: {}, predict:{}\n".format(user_id, pred))
