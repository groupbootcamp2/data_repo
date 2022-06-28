def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(file_path):
    dict = unpickle(file_path)
    return dict

import numpy as np
import  pandas as pd
base_path="C:\\Users\\user\\Desktop\\הנדסאים\\bootcamp\\project\\data\\cifar-10-python\\cifar-10-batches-py\\data_batch_"

for i in range(1,6):
    # file_path=os.path.join(base_path,str(i))
    file_path=base_path+str(i)

    # dict=np.concatenate(np.array(dict),np.array(load_data(file_path)))
# file_path="C:\\Users\\user\\Desktop\\הנדסאים\\bootcamp\\project\\data\\cifar-10-python\\cifar-10-batches-py\\data_batch_1"
print(dict)