
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
df=create_df()
print(df)
# write_df_to_csv(df)

# csv=pd.read_csv("./cifar10.csv")
# print(csv)
