'''有一个简单的方法可以检测相关性：我们用移除了某一个特征之后的数据集来构建一个监督学习（回归）模型，
然后用这个模型去预测那个被移除的特征，再对这个预测结果进行评分，看看预测结果如何。
'''
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import seaborn as sns

# TODO：为DataFrame创建一个副本，用'drop'函数丢弃一个特征
new_data = data.drop(['Frozen'], axis=1)
y = data['Frozen']

# TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size=0.2, random_state=1)

# TODO：创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

# TODO：输出在测试集上的预测得分
score = r2_score(y_test, regressor.predict(X_test), multioutput='raw_values')
print score
