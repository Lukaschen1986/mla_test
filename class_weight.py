n_samples = len(y_train)
n_classes = len(set(y_train))
y = y_train
weight_1, weight_2 = n_samples / (n_classes * np.bincount(y))
