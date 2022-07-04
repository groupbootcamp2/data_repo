import numpy as np
import pandas as pd
import os
from PIL import Image as im

# unpickle cifar 100
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# create df & images
def create_df(path=".\\resources\\", datasetType=100):
    # df = pd.DataFrame()
    dataset = "cifar" + str(datasetType)
    dirname=".\\cifar-100-python\\cifar-100-python"
    data=unpickle(os.path.join(dirname,"train"))  #data frame of data
    dictLen=len(data[b'filenames'])
    imgs=data[b'data'] #images
    newDict = {"image_name": data[b'filenames'],"image_path":[path]*dictLen, "labels": data[b'coarse_labels'], "source_image": [dataset]*dictLen, "batch":'train'}
    newDF = pd.DataFrame(newDict)
    return newDF,imgs

# reshape to 32*32*3
def reshape_images(imgs,path="",names=list()):
    print(imgs.shape)
    return imgs.reshape(len(imgs),3,32,32).transpose(0,2,3,1)
    # for arr in range(len(imgs)):
    #
    #     rgbArray = np.zeros((32, 32, 3), 'uint8')
    #     rgbArray[..., 0] = imgs[:1024].reshape(32, 32)
    #     rgbArray[..., 1] = imgs[1024:2048].reshape(32, 32)
    #     rgbArray[..., 2] = imgs[2048:3072].reshape(32, 32)
    #     img = im.fromarray(rgbArray)
    #     img.save(path)

# write data to csv file
def write_df_to_csv(df,path):
    df.to_csv(path)

# select 5 categoryes to cifar 10
def from_cifar100(df,imgs,categorys_numbers):
    newDF=pd.DataFrame()
    # newDF=df[df['labels'] ==1]
    images=np.array([])
    # imgs=reshape_images(imgs)

    # for ind in range(len(df)):#len(df)):
    #     print(ind)
    #     if df.loc[ind,'labels'] not in categorys_numbers:
    #         df.drop(ind,inplace=True)
    #         np.delete(imgs,ind
    i=0
    for ind in range(15):
        print(ind)
        if df.loc[ind,'labels'] in categorys_numbers:
            i+=1
            # pd.concat(newDF,df[ind])
            temp=pd.DataFrame(df.loc[ind])
            # newDF=newDF.append(temp)
            newDF=pd.concat([newDF,temp],axis=1)
            images=np.append(images,imgs[ind])

    print(newDF)
    print("*****************************")
    print(imgs)
    # return df,imgs
    return newDF,images

# def format_to_cifar10(data,path):
#     for arr in range(len(data)):
#         array = np.array(dict[b'images'][arr])
#         rgbArray = np.zeros((32, 32, 3), 'uint8')
#         rgbArray[..., 0] = array[:1024].reshape(32, 32)
#         rgbArray[..., 1] = array[1024:2048].reshape(32, 32)
#         rgbArray[..., 2] = array[2048:3072].reshape(32, 32)
#         img = im.fromarray(rgbArray)
#         img.save(path + data.iloc[arr]["images"])

# save images in computer
def save_images(images,path,names):
    images=reshape_images(images,path=path)
    for image in images:
        img = im.fromarray(images[image])
        img.save(path + names[image])

# write data to csv file
def write_df_to_csv(df):
    df.to_csv(".\\cifar100.csv")

def read_csv(path):
    return pd.read_csv(path)

def concat_cifar10_cifar100(cifar10_path,cifar100_path,concat_path):
    cifar10=read_csv(cifar10_path)
    cifar100=read_csv(cifar100_path)
    write_df_to_csv(pd.concat([cifar10,cifar100]),concat_path)

#  df & images array - load data - main function
def load_cifar_100():
    path = ".\\resources"
    df, imgs = create_df()
    # the selected classes
    df,imgs = from_cifar100(df,imgs, [1, 2, 3, 4, 5])
    save_images(imgs,path,df.loc["image_name"])
    print(df)

    # write to csv file
    write_df_to_csv(df,".\\cifar100.csv")

#     concat cifar10_cifar100
    concat_cifar10_cifar100(".\\cifar10.csv",".\\cifar100.csv",".\\concat.csv")



load_cifar_100()


# TODO:
# transfer cifar100 csv titles to cifar10 csv titles   V?
# concat cifar 10 & cifar 100  V
# split code to classes or modules by roles
# train - test - validation


