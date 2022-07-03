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
def images(imgs,path="",names=list()):
    print(imgs.shape)
    return  imgs.reshape(len(imgs),3,32,32).transpose(0,2,3,1)
    # for arr in range(len(imgs)):
    #
    #     rgbArray = np.zeros((32, 32, 3), 'uint8')
    #     rgbArray[..., 0] = imgs[:1024].reshape(32, 32)
    #     rgbArray[..., 1] = imgs[1024:2048].reshape(32, 32)
    #     rgbArray[..., 2] = imgs[2048:3072].reshape(32, 32)
    #     img = im.fromarray(rgbArray)
    #     img.save(path)


# def save_images(images):


def write_df_to_csv(df):
    df.to_csv("./cifar10.csv")

def from_cifar100(df,imgs,categorys_numbers):
    indexes_to_drop=[]
    imgs=images(imgs)
    for ind in range(len(df)):#len(df)):
        print(ind)
        if df.loc[ind,'labels'] not in categorys_numbers:
            df.drop(ind,inplace=True)
            np.delete(imgs,ind)



    return df,imgs

# def format_to_cifar10(data,path):
#     for arr in range(len(data)):
#         array = np.array(dict[b'images'][arr])
#         rgbArray = np.zeros((32, 32, 3), 'uint8')
#         rgbArray[..., 0] = array[:1024].reshape(32, 32)
#         rgbArray[..., 1] = array[1024:2048].reshape(32, 32)
#         rgbArray[..., 2] = array[2048:3072].reshape(32, 32)
#         img = im.fromarray(rgbArray)
#         img.save(path + data.iloc[arr]["images"])

def write_df_to_csv(df):
    df.to_csv(".\\cifar100.csv")

#  df & images array
def load_cifar_100():
    df, imgs = create_df()
    imgs = images(imgs)
    # the selected classes
    df,imgs = from_cifar100(df,imgs, [1, 2, 3, 4, 5])
    print(df)
    path = ".\\resources"

    # write to csv file
    write_df_to_csv(df)



load_cifar_100()


# todo:
# transfer cifar100 csv titles to cifar10 csv titles
# concat cifar 10 & cifar 100

