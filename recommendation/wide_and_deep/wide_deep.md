# Wide & Deep

> 原始论文：[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)




## 参考
- [github:Wide&Deep模型解读](https://github.com/ShaoQiBNU/wide_and_deep)
- [csdn: Wide&Deep翻译](https://blog.csdn.net/starzhou/article/details/78845931)





**提出了一个结合使用了非线性特征的线性模型和一个用来embedding特征的深度学习，并且使用联合训练的方法进行优化。思想是，基于交叉特征的线性模型只能从历史出现过的数据中找到非线性（显性的非线性），深度学习可以找到没有出现过的非线性（隐形的非线性）。**

**这里我没有对memorization和generalization进行翻译，因为我也不知道怎么翻译，memorization的话就是去把历史数据中显性的非线性找出来，generalization的就是范化性，就是把一些隐性的找出来。**





## 物理意义

组合特征：逻辑回归，
深层特诊



# 逻辑回归和深度模型



### w&d物理意义：

1. 为何优于逻辑回归和DNN深度神经网络模型？

推荐系统两个概念：泛化和记忆

泛化：推荐的多样性

记忆：

例子：某人一直看宫斗类型电视剧，推荐系统会一直给他推宫斗类型的电视剧，这就是记忆。但用户有可能喜欢看历史类型的电视剧，如何推荐历史类型的电视剧



2. 如何增加泛化能力？

单用逻辑回归：需要人工组合类型，如果训练数据未出现历史观看特征无法组合发现

DNN：基础特征组合，在隐藏层学习出高位特征。可能过泛化



泛化和记忆



记忆：

同一类型；

- 逻辑回归

泛化：

- 逻辑回归需要人工组合类型
- DNN深度神经网络可组合高维度特征



逻辑回归记忆、泛化差

DNN可能会过泛化

### w&d网络结构

wide：将离散特征组合；加权和

deep：全连接网络

DNN的最后一个隐层与逻辑回归输出加和后经过一个激活函数得到输出

可以保证反向传播同时更新两侧参数；联合训练



W:离散特征及离散组合放入w侧

D:连续特征和字符型特征hash embedding后传入d侧



### w&d数学原理：



#### 模型输出



$$
a_{out}^T=h(w_{wide},w_{deep})=\sigma(w_{wide}[x,x_{cross}]+w_{deep}a_{out}^{T-1}+b^T)
$$

反向传播如何进行？

#### Wide侧参数学习：

$$
\frac{\partial L}{\partial w_{wide_j}}= \frac{\partial L}{\partial a^T}\frac{\partial a^T}{\partial z^T} \frac{\partial z^T}{\partial w_{wide_j}}=\frac{\partial L}{\partial a^T}\sigma^\prime(z^T)x_{wide_j}
$$

$$
其中： z_j ^t= \sum_{k=1}^n w_{jk}^ta_k^{t-1}+b_j^t， a^t=\sigma(z^t) 激活函数
$$



$$
L=(y_j^t-a_j^t)^2, \frac {\partial L}{\partial a_j^t}, \frac {\partial L}{\partial z_j^t}=\frac{\partial L}{\partial a_j^t}\frac{\partial a_j^t}{\partial z_j^t}=\frac {\partial L}{\partial a_j^T}\sigma^\prime(z^t)
$$


L：loss函数，a：神经元激活值，w：参数，z：某一层所有激活前。w_jk^t：t-1层第k节点指向t层j节点

最终激活函数：阶跃函数,sgn

Deep激活：修正线性单元 Relu

隐层之间全连接：全连接相当于逻辑回归的特征交叉

优化器：**FTRL**

#### deep侧参数学习： 

$$
\frac{\partial L}{\partial z_j^{t-1}}=\sum_{k=1}^n \frac{\partial L}{\partial z_k^t} \frac{\partial z_k^t}{\partial a_j^{t-1}} \frac{\partial a_j^{t-1}}{\partial z_j^{t-1}}=\sum_k \frac {\partial L}{\partial z_k^t} w_{deep_{kj}}^t \frac{\partial a_j^{t-1}}{\partial z_j^{t-1}}=\sum_k \frac {\partial L}{\partial z_k^t} w_{deep_{kj}}^t \sigma^\prime(z^{t-1})
$$

$$
中间层：           z_k^t = \sum_j w_{deep_{kj}}^ja_j^{t-1}+b_k^t \rightarrow t\neq T
$$

$$
最后一层：z_k^t = (\sum_j w_{deep_{kj}}^ja_j^{t-1}+b_k^t)+w_{wide}*X \rightarrow t= T
$$

T：最后一层,w_k^j：t-1层第k节点指向t层j节点

损失函数对任一层z的偏导
$$
\frac{\partial L}{\partial b_j^t}=\frac{\partial L}{\partial z_j^t} \frac{\partial z_j^t}{\partial b_j^{t-1}}= \frac{\partial L}{\partial z_j^t}
$$

$$
\frac{\partial L}{\partial w_{jk}^t}=\frac{\partial L}{\partial z_j^t} \frac{\partial z_j^t}{\partial w_{jk}^{t-1}}= \frac{\partial L}{\partial z_j^t} a_k^{t-1}
$$





优化器：**AdaGrad**





### 优缺点





优缺点

缺点：Wide 部分还是需要人为的特征工程。优点：实现了对 memorization 和 generalization 的统一建模



假设与不足







