import numpy as np
import pandas as pd
import os
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_df():
    df = pd.DataFrame()
    dirname="C:\\Users\\User\\Downloads\\cifar-10-python\\cifar-10-batches-py"
    for i in range(1,6):
        file=os.path.join(dirname,"data_batch_"+str(i))
        dict=unpickle(file)
        DICTLEN=len(dict[ b'filenames'])
        newDict={"images":dict[ b'filenames'],"labels": dict[b'labels'] , "source_image": ["cifar10"]*DICTLEN, "batch":[i]*DICTLEN}
        newDF=pd.DataFrame(newDict)
        df=pd.concat([df,newDF])
    return df


def write_df_to_csv(df):
    df.to_csv("C:\\Users\\User\\Downloads\\cifar10.csv")
df=create_df()
write_df_to_csv(df)

csv=pd.read_csv("C:\\Users\\User\\Downloads\\cifar10.csv")
print(csv)
