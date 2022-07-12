import numpy as np
import pandas as pd
import os
from PIL import Image as im
from typing import List
from sklearn.utils import shuffle
import glob
import cv2

CSV_PATH = "..\\cifar10_100.csv"


def create_classes_dict():
    meta_dict10 = unpickle("..\\cifar-10\\batches.meta")
    classes_dict = {np.arange(0, 10)[i]: (str(meta_dict10[b'label_names'][i]))[2:-1] for i in range(10)}
    meta_dict100 = unpickle("..\\cifar-100\\meta")
    classes_dict100 = {np.arange(10, 25)[i]: (str(meta_dict100[b'coarse_label_names'][i]))[2:-1] for i in range(15)}
    classes_dict.update(classes_dict100)
    return classes_dict


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_df(datasetType: int, files_list: List[str], labels: str):
    df = pd.DataFrame()
    image_data = [[0] * 3072]
    dataset = f"cifar {datasetType}"
    dirname = f"..\\cifar-{datasetType}"
    path = "..\\resources\\"
    for i in range(len(files_list)):
        file = os.path.join(dirname, files_list[i])
        dict = unpickle(file)
        DictLen = len(dict[b'filenames'])
        newDict = {"image_name": dict[b'filenames'], "image_path": [path] * DictLen,
                   "source_image": [dataset] * DictLen, "batch": [files_list[i]] * DictLen, "label": dict[labels]}
        newDict["image_name"] = [(str(sub)).replace("'", "")[1:] for sub in newDict["image_name"]]
        newDF = pd.DataFrame(newDict)
        df = pd.concat([df, newDF])
        image_data.extend(dict[b'data'])
    return df, image_data[1:]


def save_images(df, data):
    for arr in range(len(data)):
        print(arr)
        array = np.array(data[arr])
        rgbArray = np.zeros((32, 32, 3), 'uint8')
        rgbArray[..., 0] = array[:1024].reshape(32, 32)
        rgbArray[..., 1] = array[1024:2048].reshape(32, 32)
        rgbArray[..., 2] = array[2048:3072].reshape(32, 32)
        img = im.fromarray(rgbArray)
        img.save(df.iloc[arr]["image_path"] + df.iloc[arr]["image_name"])


def choose_x_labels_from_cifar100(df, imgs, categorys_numbers, old_csv):
    images = [[0] * 3072]
    df['index'] = [i for i in range(len(df['label']))]
    df['label'] = [label + 10 for label in df["label"]]
    categorys_numbers = categorys_numbers + 10
    print(categorys_numbers)
    df = df[(df['label'].isin(categorys_numbers)) & (~df['label'].isin(old_csv['label'].value_counts().keys()))]
    for i in df['index']:
        images.append(imgs[i])
    df.drop('index', inplace=True, axis=1)
    return df, images[1:]


# def write_df_to_csv(df,path=CSV_PATH, newFile=False):
#     if newFile:
#         df.to_csv(path)
#     else:
#         try:
#             with open(path, 'a') as fd:
#                 fd.writelines(df)
#         except:
#             print("except from write df to csv" )
#             df.to_csv(path)


def write_df_to_csv(df, path=CSV_PATH, newFile=False):
    if newFile:
        if (os.path.exists(newFile) and os.path.isfile(newFile)):
            os.remove(newFile)
    df.to_csv(path,mode='a')


# def read_csv11(path=CSV_PATH):
#     try:
#         with open(path, 'r') as fd:
#             return fd.read()
#     except:
#         return "no file"


def load_cifar10():
    df, image_data = create_df(10, [('data_batch_' + str(i)) for i in range(1, 6)] + ["test_batch"], b'labels')
    write_df_to_csv(df, newFile=True)
    save_images(df, image_data)
    return df


def load_x_labels_from_cifar100(categorys_numbers: List[int]):
    old_csv = pd.DataFrame(pd.read_csv(CSV_PATH))
    cifar100, image_data = create_df(100, ['train', 'test'], b'coarse_labels')
    cifar100, image_data = choose_x_labels_from_cifar100(cifar100, image_data, np.array(categorys_numbers), old_csv)

    write_df_to_csv(cifar100)
    save_images(cifar100, image_data)
    return cifar100


# 11odo: genery precent and add owner images to test
def split_train_test_validation(path=CSV_PATH):
    # read data from csv and split cifar10 and cifar100
    cifar10_100 = pd.read_csv(path)
    cifar10 = cifar10_100[cifar10_100['source_image'] == 'cifar 10']
    cifar100 = cifar10_100[cifar10_100['source_image'] == 'cifar 100']
    cifar100 = shuffle(cifar100)
    len_cifar_10 = len(cifar10)
    len_cifar_100 = len(cifar100)
    # split cifar10 to train validation test
    x_train = cifar10.iloc[:int(0.6 * len_cifar_10), :-1]
    y_train = cifar10.iloc[:int(0.6 * len_cifar_10), -1]
    x_validation = cifar10.iloc[int(0.6 * len_cifar_10):int(0.8 * len_cifar_10), :-1]
    y_validation = cifar10.iloc[int(0.6 * len_cifar_10):int(0.8 * len_cifar_10), -1]
    x_test = cifar10.iloc[int(0.8 * len_cifar_10):, :-1]
    y_test = cifar10.iloc[int(0.8 * len_cifar_10):, -1]
    # split cifar100 to train validation test
    x_train = pd.concat([x_train, cifar100[cifar100['batch'] == 'train'].iloc[:int(0.6 * len_cifar_100), :-1]])
    y_train = pd.concat([y_train, cifar100.iloc[:int(0.6 * len_cifar_100), -1]])
    x_validation = pd.concat([x_validation, cifar100.iloc[int(0.6 * len_cifar_100):int(0.8 * len_cifar_100), :-1]])
    y_validation = pd.concat([y_validation, cifar100.iloc[int(0.6 * len_cifar_100):int(0.8 * len_cifar_100), -1]])
    x_test = pd.concat([x_test, cifar100.iloc[int(0.8 * len_cifar_100):, :-1]])
    y_test = pd.concat([y_test, cifar100.iloc[int(0.8 * len_cifar_100):, -1]])

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def save_images_as_nparray():
    Xtrain, Xvalidation, Xtest, Ytrain, Yvalidation, Ytest = split_train_test_validation()

    images_Xtrain = [cv2.imread(Xtrain.iloc[i]['image_path'] + Xtrain.iloc[i]["image_name"]) for i in range(10)]
    images_Xvalidation = [cv2.imread(Xvalidation.iloc[i]['image_path'] + Xvalidation.iloc[i]["image_name"]) for i in
                          range(10)]
    images_Xtest = [cv2.imread(Xtest.iloc[i]['image_path'] + Xtest.iloc[i]["image_name"]) for i in range(10)]

    images_Xtrain = np.array((images_Xtrain))
    images_Xvalidation = np.array((images_Xvalidation))
    images_Xtest = np.array((images_Xtest))
    Ytrain = np.array((Ytrain))
    Yvalidation = np.array((Yvalidation))
    Ytest = np.array((Ytest))
    return images_Xtrain, images_Xvalidation, images_Xtest, Ytrain, Yvalidation, Ytest


def save_to_zip():
    images_Xtrain, images_Xvalidation, images_Xtest, Ytrain, Yvalidation, Ytest = save_images_as_nparray()
    np.savez('cfar10_modified_1000.npz', train=images_Xtrain, ytrain=Ytrain, validation=images_Xvalidation,
             yvalidation=Yvalidation, test=images_Xtest, ytest=Ytest)
