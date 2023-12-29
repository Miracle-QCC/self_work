import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# 加载 iris 数据集
iris = load_iris()
X, y = iris.data, iris.target

# 选择模型
model = RandomForestClassifier()

# 使用 RFECV 进行特征选择
cv = StratifiedKFold(10)  # 使用分层交叉验证
selector = RFECV(estimator=model, step=1, cv=cv)
selector = selector.fit(X, y)

# 查看结果
print("Optimal number of features : %d" % selector.n_features_)
print("Ranking of features : %s" % selector.ranking_)

# 绘制折线图
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
plt.show()
