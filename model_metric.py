fig, ax = plt.subplots(figsize=(8,6))
plt.scatter(x=, y=, alpha=0.5)
plt.xlabel("")
plt.ylabel("")

from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    score = r2_score(y_true, y_predict)
    return score
    
def performance_metric2(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    SSE = np.sum((y_true-y_predict)**2)
    SST = np.sum((y_true-np.mean(y_true))**2)
    score = 1 - SSE / SST
    return score

