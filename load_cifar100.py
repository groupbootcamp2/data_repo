
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
    # for i in range(1,6):
    #     file=os.path.join(dirname,"data_batch_"+str(i))
    #     dict=unpickle(file)
    #     DICTLEN=len(dict[ b'filenames'])
    #     newDict={"images":dict[ b'filenames'],"labels": dict[b'labels'] , "source_image": ["cifar10"]*DICTLEN, "batch":[i]*DICTLEN}
    #     newDF=pd.DataFrame(newDict)
    #     df=pd.concat([df,newDF])
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
