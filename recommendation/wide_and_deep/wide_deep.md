# Wide & Deep

> 原始论文：[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)




## 参考
- [github:Wide&Deep模型解读](https://github.com/ShaoQiBNU/wide_and_deep)
- [csdn: Wide&Deep翻译](https://blog.csdn.net/starzhou/article/details/78845931)
- [计算广告CTR预估系列(四)--Wide&Deep理论与实践](<https://blog.csdn.net/u010352603/article/details/80590129>)



点击率预估：

分类模型：



**提出了一个结合使用了非线性特征的线性模型和一个用来embedding特征的深度学习，并且使用联合训练的方法进行优化。思想是，基于交叉特征的线性模型只能从历史出现过的数据中找到非线性（显性的非线性），深度学习可以找到没有出现过的非线性（隐形的非线性）。**

**这里我没有对memorization和generalization进行翻译，因为我也不知道怎么翻译，memorization的话就是去把历史数据中显性的非线性找出来，generalization的就是范化性，就是把一些隐性的找出来。**





## 物理意义

组合特征：逻辑回归，
深层特诊



# 逻辑回归和深度模型



论文内容：

wide&deep优点，网络结构和对比结果



### w&d物理意义：

1. 为何优于逻辑回归和DNN深度神经网络模型？

推荐系统两个概念：泛化和记忆

泛化：推荐的多样性；发现

记忆：推荐的相似性；历史

例子：某人一直看宫斗类型电视剧，推荐系统会一直给他推宫斗类型的电视剧，这就是记忆。但用户有可能喜欢看历史类型的电视剧，如何推荐出历史类型的电视剧



2. 如何增加泛化能力？

单用逻辑回归：需要人工组合类型，如果训练数据未出现历史观看特征无法组合发现；但是深度神经网络DNN可以将低维特征进行高维组合，在隐层中学到一些高维特征

DNN：基础特征组合，在隐藏层学习出高维特征。

问题：如果用户历史行为数据不充分，可能过泛化；造成推荐结果大多不相关



wide&deep模型可以平衡这两个方面



泛化和记忆



记忆：

同一类型；

- 逻辑回归

泛化：

- 逻辑回归需要人工组合类型
- DNN深度神经网络可组合高维度特征



DNN可能会过泛化

FM和DNN都算是这样的模型，可以在很少的特征工程情况下，通过学习一个低纬度的embedding vector来学习训练集中从未见过的组合特征。

FM和DNN的缺点在于：当query-item矩阵是稀疏并且是high-rank的时候（比如user有特殊的爱好，或item比较小众），很难非常效率的学习出低维度的表示。这种情况下，大部分的query-item都没有什么关系。但是dense embedding会导致几乎所有的query-item预测值都是非0的，这就导致了推荐过度泛化，会推荐一些不那么相关的物品。

相反，linear model却可以通过cross-product transformation来记住这些exception rules，而且仅仅使用了非常少的参数。

总结一下：

线性模型无法学习到训练集中未出现的组合特征； 易于理解，计算代价小。容易欠拟合，未学习到交叉规律，需要手动交叉特征，手动构造特征

FM或DNN通过学习embedding vector虽然可以学习到训练集中未出现的组合特征，但是会过度泛化。推荐难以理解。

Wide & Deep Model通过组合这两部分，解决了这些问题。

**对于低阶的组合特征，学习到的比较少。**

### w&d网络结构

![](image/image01.png)

wide：逻辑回归，将离散特征组合；加权和

deep：深度神经网络，全连接网络

DNN的最后一个隐层与逻辑回归输出加和后经过一个激活函数得到输出

这个结构可以保证每次反向传播同时更新两侧参数；达到联合训练

特征处理：

W:离散特征及离散组合放入w侧

D:连续特征和字符型特征hash embedding后传入d侧



### w&d数学原理：



#### 模型输出



$$
a_{out}^T=h(w_{wide},w_{deep})=\sigma(w_{wide}[x,x_{cross}]+w_{deep}a_{out}^{T-1}+b^T)
$$

#### Wide侧参数学习：

$$
\frac{\partial L}{\partial w_{wide_j}}= \frac{\partial L}{\partial a^T}\frac{\partial a^T}{\partial z^T} \frac{\partial z^T}{\partial w_{wide_j}}=\frac{\partial L}{\partial a^T}\sigma^\prime(z^T)x_{wide_j}
$$

$$
其中： z_j ^t= \sum_{k=1}^n w_{jk}^ta_k^{t-1}+b_j^t， a_j^t=\sigma(z_j^t) 激活函数
$$

二次损失函数为例：

$$
L=(y_j^t-a_j^t)^2, \frac {\partial L}{\partial a_j^t}=-2\partial a_j^t, \frac {\partial L}{\partial z_j^t}=\frac{\partial L}{\partial a_j^t}\frac{\partial a_j^t}{\partial z_j^t}=\frac {\partial L}{\partial a_j^t}\sigma^\prime(z^t)
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
中间层：           z_k^t = \sum_j w_{deep_{kj}}^t a_j^{t-1}+b_k^t \rightarrow t\neq T
$$

$$
最后一层：z_k^t = (\sum_j w_{deep_{kj}}^ja_j^{t-1}+b_k^t)+w_{wide}*X \rightarrow t= T
$$

T：最后一层,w_k^j：t-1层第k节点指向t层j节点

损失函数对任一层z的偏导
$$
\frac{\partial L}{\partial b_j^t}=\frac{\partial L}{\partial z_j^t} \frac{\partial z_j^t}{\partial b_j^t}= \frac{\partial L}{\partial z_j^t}
$$

$$
\frac{\partial L}{\partial w_{jk}^t}=\frac{\partial L}{\partial z_j^t} \frac{\partial z_j^t}{\partial w_{jk}^t}= \frac{\partial L}{\partial z_j^t} a_k^{t-1}
$$





优化器：**AdaGrad**





### 优缺点



一般用于大规模带有稀疏输入的回归和分类问题，如推荐系统、搜索、排名等问题

输入稀疏通常是由离散特征有非常非常多个可能的取值造成的，one-hot之后维度非常大。

## 5. 优缺点

缺点：Wide部分还是需要人为的特征工程。 
优点：实现了对memorization和generalization的统一建模。



多种类型的叉乘特征变换记忆特定的特征组合；难以归纳出没有出现过的特征组合，这需要人工提供特征

引入深度深刻网络，可以通过低维嵌入更好的归纳，推荐出相似类型的物品；但有时候记录特定的组合作为规则和例外是非常重要的；泛化的相似物品用户不一定喜欢





优缺点

缺点：Wide 部分还是需要人为的特征工程。优点：实现了对 memorization 和 generalization 的统一建模

1. 偏向于提取低阶或者高阶的组合特征。不能同时提取这两种类型的特征。
2. 需要专业的领域知识来做特征工程。

假设与不足

**Wide&Deep缺点：**

1. 需要特征工程提取低阶组合特征

#### DeepFM

**优点：**

1. 没有用FM去预训练隐向量V，并用V去初始化神经网络。（相比之下FNN就需要预训练FM来初始化DNN）
2. FM模块不是独立的，是跟整个模型一起训练学习得到的。（相比之下Wide&Deep中的Wide和Deep部分是没有共享的）
3. 不需要特征工程。（相比之下Wide&Deep中的Wide部分需要特征工程）
4. 训练效率高。（相比PNN没有那么多参数）

**Wide&Deep缺点：**

1. 需要特征工程提取低阶组合特征

#### DeepFM

**优点：**

1. 没有用FM去预训练隐向量V，并用V去初始化神经网络。（相比之下FNN就需要预训练FM来初始化DNN）
2. FM模块不是独立的，是跟整个模型一起训练学习得到的。（相比之下Wide&Deep中的Wide和Deep部分是没有共享的）
3. 不需要特征工程。（相比之下Wide&Deep中的Wide部分需要特征工程）
4. 训练效率高。（相比PNN没有那么多参数）



### 模型实现

tensorflow

官网给出过实现给出了一个代码示例<https://github.com/tensorflow/models/tree/master/official/wide_deep>

数据源： [UCI开源数据集Adult](http://archive.ics.uci.edu/ml/datasets/Adult?spm=a2c4e.11153940.blogcont53862.9.6ee61e33rzTKQl) 
针对美国某区域的一次人口普查结果，共32561条数据。具体字段如下表：

| 字段名          | 含义           | 类型   |
| :-------------- | :------------- | :----- |
| age             | 年龄           | double |
| workclass       | 工作类型       | string |
| fnlwgt          | 序号           | string |
| education       | 教育程度       | string |
| education_num   | 受教育时间     | double |
| maritial_status | 婚姻状况       | string |
| occupation      | 职业           | string |
| relationship    | 关系           | string |
| race            | 种族           | string |
| sex             | 性别           | string |
| capital_gain    | 资本收益       | string |
| capital_loss    | 资本损失       | string |
| hours_per_week  | 每周工作小时数 | double |
| native_country  | 原籍           | string |
| income          | 收入           | string |

收入预测

连续特征：

大规模数据集上wd模型表现更好





