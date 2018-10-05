def prior_prob(df_train, y_name, y_label):
    count = df_train[y_name].value_counts()[y_label]
    res = count/len(df_train)
    return res
    
def condi_prob(df_train, y_name, y_label, df_predict):
    df_sub = df_train[df_train[y_name] == y_label]
    n = len(df_predict); p = len(df_predict.columns)
    probs = np.zeros((n,1))
    for i in range(n):
        for j in range(p):
            # 如果是not_na_jnt，并且是yes，以及False+vip_price， 给予3倍加成，否则/3
            if isinstance(df_predict.iloc[i,j], np.str) and df_predict.columns[j] == "not_na_jnt" and y_label == "yes":
                if df_predict.iloc[i,j] == "False+vip_price":
                    probs[i] += np.log( (np.sum(df_predict.iloc[i,j] == df_sub.iloc[:,j])+1)/(len(df_sub)+p)*3 )
                else:
                    probs[i] += np.log( (np.sum(df_predict.iloc[i,j] == df_sub.iloc[:,j])+1)/(len(df_sub)+p)/3 )
            # 如果是is_equal_jnt，并且是yes，以及False+private， 给予2倍加成，否则/2
            elif isinstance(df_predict.iloc[i,j], np.str) and df_predict.columns[j] == "is_equal_jnt" and y_label == "yes":
                if df_predict.iloc[i,j] == "False+private":
                    probs[i] += np.log( (np.sum(df_predict.iloc[i,j] == df_sub.iloc[:,j])+1)/(len(df_sub)+p)*2 )
                else:
                    probs[i] += np.log( (np.sum(df_predict.iloc[i,j] == df_sub.iloc[:,j])+1)/(len(df_sub)+p)/2 )
            # 如果是int64，且<=3和yes，则/3，否则不变
            elif isinstance(df_predict.iloc[i,j], np.int64):
                if df_predict.iloc[i,j] <= 3 and y_label == "yes":
                    probs[i] += np.log( (stats.poisson.pmf(df_predict.iloc[i,j], np.mean(df_sub.iloc[:,j]))+0.0001)/3 )
                else:
                    probs[i] += np.log( stats.poisson.pmf(df_predict.iloc[i,j], np.mean(df_sub.iloc[:,j]))+0.0001 )
            else:
                probs[i] += np.log( (np.sum(df_predict.iloc[i,j] == df_sub.iloc[:,j])+1)/(len(df_sub)+p) )
    probs = np.exp(probs)
    return probs

def predict_func(df_train, y_name, y_label_set, df_predict, alpha):
    n = len(df_predict)
    pred = np.zeros((n,2))
    pq = []
    for i in range(len(y_label_set)):
        condi_probs = condi_prob(df_train, y_name, y_label_set[i], df_predict) # 条件概率
        prior_probs = prior_prob(df_train, y_name, y_label_set[i]) # 先验概率
        pq.append(prior_probs) # 储存先验概率
        joint_prob = condi_probs * prior_probs
        pred[:,i][:,np.newaxis] = joint_prob
    KL = pq[0] * np.log(pq[0]/pq[1]) # 计算q相对于p的信息损失
    para = np.exp(KL+0.001)**alpha # 转化为para
    pred[:,0] /= para # 系数修正
    pred[:,1] *= para # 系数修正
    posterior_prob = pred/np.sum(pred, axis=1, keepdims=True) # 后验概率
    posterior_res = pd.DataFrame(posterior_prob, columns=y_label_set).round(4)
    posterior_res["y_pred"] = posterior_res.columns[np.argmax(posterior_prob, axis=1)]
    return posterior_res
