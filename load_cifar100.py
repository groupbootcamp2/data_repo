import numpy as np
import pandas as pd
import os
from PIL import Image as im

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# create df & reshape images
def create_df():
    # df = pd.DataFrame()
    dirname=".\\cifar-100-python\\cifar-100-python"
    data=unpickle(os.path.join(dirname,"train"))
    imgs=data[b'data']
    imgs=images(imgs,path=".\\resources",names=data[b'filenames'])
    newDict = {"images": data[b'filenames'], "labels": data[b'coarse_labels']}
    newDF = pd.DataFrame(newDict)
    return newDF,imgs

# reshape to 32*32*3
def images(imgs,path,names):
    return  imgs.reshape(len(imgs),3,32,32).transpose(0,2,3,1)

def write_df_to_csv(df):
    df.to_csv("./cifar10.csv")

def from_cifar100(df,categorys_numbers):
    indexes_to_drop=[]
    for ind in range(len(df)):
        if df.loc[ind,'labels'] not in categorys_numbers:
            df.drop(ind,inplace=True)
    return df

# def format_to_cifar10(data,path):
#     for arr in range(len(data)):
#         array = np.array(dict[b'images'][arr])
#         rgbArray = np.zeros((32, 32, 3), 'uint8')
#         rgbArray[..., 0] = array[:1024].reshape(32, 32)
#         rgbArray[..., 1] = array[1024:2048].reshape(32, 32)
#         rgbArray[..., 2] = array[2048:3072].reshape(32, 32)
#         img = im.fromarray(rgbArray)
#         img.save(path + data.iloc[arr]["images"])

#  df & images array
df,imgs=create_df()

# the selected classes
df=from_cifar100(df,[1,2,3,4,5])

print(df)
path=".\\resources"

# write to csv file
write_df_to_csv(df)


# todo:
# * Load only the images - the data set obtained from the create df () function from the classes we selected



def write_df_to_csv(df):
    df.to_csv("..\\cifar10.csv")
