
import pandas as pd
import os
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_df():
    # df = pd.DataFrame()
    dirname=".\\cifar-100-python\\cifar-100-python"
    dict=unpickle(os.path.join(dirname,"train"))
    DICTLEN = len(dict[b'filenames'])
    newDict = {"images": dict[b'filenames'], "labels": dict[b'coarse_labels']}
    newDF = pd.DataFrame(newDict)
    return newDF


def write_df_to_csv(df):
    df.to_csv("./cifar10.csv")

def from_cifar100(df,categorys_numbers):
    indexes_to_drop=[]
    for ind in range(len(df)):
        if df.loc[ind,'labels'] not in categorys_numbers:
            df.drop(ind,inplace=True)
    return df

df=create_df()
# print(df)
df=from_cifar100(df,[1,2,3,4,5])
print(df)

def write_df_to_csv(df):
    df.to_csv("..\\cifar10.csv")
