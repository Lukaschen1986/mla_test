def layer_func(p, outNum, layer_num, neuron_num):
    w = []; b = []
    middle_num = layer_num-2
    for i in range(middle_num):
        if middle_num < 0:
            print("Error: layer must >= 2 !")
            break
    
    
    w_first = np.random.normal(0, 0.01, p*neuron_num).reshape(p,neuron_num); w.append(w_first)
    w_last = np.random.normal(0, 0.01, neuron_num*outNum).reshape(neuron_num,outNum); w.append(w_last)
    b_first = np.zeros((1, neuron_num)); b.append(b_first)
    b_last = np.zeros((1, outNum)); b.append(b_last)
    middle_num = layer_num-2
    for i in range(middle_num):
        if middle_num < 0:
            print("Error: layer must >= 2 !")
            break
        elif middle_num == 0:
            return w, b
        else:
            w_middle = np.random.normal(0, 0.01, neuron_num*neuron_num).reshape(neuron_num,neuron_num)
            b_middle = np.zeros((1, neuron_num))
            w.append(w_middle)
            b.append(b_middle)
    return w, b
layer_func(p=5, outNum=10, layer_num=1, neuron_num=4)
