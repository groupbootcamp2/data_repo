import numpy as np
import pandas as pd
import os
from PIL import Image as im
from typing import List
from sklearn.utils import shuffle
import cv2
from PIL import Image
import config as cnfg



def create_classes_dict():
    meta_dict10 = unpickle(f"..\\{cnfg.cifar10}\\{cnfg.meta_file_cifar10}")
    classes_dict = {np.arange(0, cnfg.num_classes_cifar10)[i]: (str(meta_dict10[b'label_names'][i]))[2:-1] for i in range(10)}
    meta_dict100 = unpickle(f"..\\{cnfg.cifar100}\\{cnfg.meta_file_cifar100}")
    classes_dict100 = {np.arange(cnfg.num_classes_cifar10, cnfg.num_classes_cifar10+cnfg.num_classes_cifar100)[i]: (str(meta_dict100[b'coarse_label_names'][i]))[2:-1] for i in range(15)}
    classes_dict.update(classes_dict100)
    return classes_dict


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_df(dataset_dirname: str, files_list: List[str], labels: str):
    df = pd.DataFrame()
    image_data = [[0] * 3072]
    for i in range(len(files_list)):
        file = os.path.join(dataset_dirname, files_list[i])
        dict = unpickle(file)
        DictLen = len(dict[b'filenames'])
        newDict = {"image_name": dict[b'filenames'], "image_path": [cnfg.images_dir_path] * DictLen,
                   "source_image": [dataset_dirname] * DictLen, "batch": [files_list[i]] * DictLen, "label": dict[labels]}
        newDict["image_name"] = [(str(sub)).replace("'", "")[1:] for sub in newDict["image_name"]]
        newDF = pd.DataFrame(newDict)
        df = pd.concat([df, newDF])
        image_data.extend(dict[b'data'])
    return df, image_data[1:]

def save_image( data_arr, path_to_save):
    array = np.array(data_arr)
    vector_size=cnfg.image_size*cnfg.image_size #vector_size_for_each_color
    rgbArray = np.zeros((cnfg.image_size, cnfg.image_size, 3), 'uint8')
    rgbArray[..., 0] = array[:vector_size].reshape(cnfg.image_size, cnfg.image_size)
    rgbArray[..., 1] = array[vector_size:vector_size*2].reshape(cnfg.image_size, cnfg.image_size)
    rgbArray[..., 2] = array[vector_size*2:vector_size*3].reshape(cnfg.image_size, cnfg.image_size)
    img = im.fromarray(rgbArray)
    img.save(path_to_save)


def save_images(df, data):
    for arr in range(len(data)):
        print(arr)
        save_image(data[arr],df.iloc[arr]["image_path"] + df.iloc[arr]["image_name"])


def choose_x_labels_from_cifar100(df, imgs, categorys_numbers, old_csv):
    images = [[0] * cnfg.image_size*cnfg.image_size*3]
    df['index'] = [i for i in range(len(df['label']))]
    df['label'] = [label + cnfg.num_classes_cifar10 for label in df["label"]]
    categorys_numbers = categorys_numbers + cnfg.num_classes_cifar10
    df = df[(df['label'].isin(categorys_numbers)) & (~df['label'].isin(old_csv['label'].value_counts().keys()))]
    for i in df['index']:
        images.append(imgs[i])
    df.drop('index', inplace=True, axis=1)
    return df, images[1:]




def insert_personal_image_to_csv (image_name:str, label:int ):
    df = pd.DataFrame({"image_name": [image_name], "image_path": [cnfg.images_directory_path], "source_image":['personal'], "batch": ['test'], "label": [label]})
    write_df_to_csv(df)
    im = Image.open(cnfg.personal_image_path)
    im_resize=im.resize((cnfg.image_size,cnfg.image_size))
    im_resize.save(df.iloc[0]["image_path"] + df.iloc[0]["image_name"])





def write_df_to_csv(df, newFile=False):
    if newFile:
        df.to_csv(cnfg.csv_path, index=False)
    else:
        df.to_csv(cnfg.csv_path, mode='a', index=False, header=False)


def load_cifar10():
    df, image_data = create_df(cnfg.cifar10, cnfg.files_list_cifar10, cnfg.label_head_cifar10)
    write_df_to_csv(df, newFile=True)
    #save_images(df, image_data)
    return df


def load_x_labels_from_cifar100(categorys_numbers: List[int]):
    old_csv = pd.DataFrame(pd.read_csv(cnfg.csv_path))
    cifar100, image_data = create_df(cnfg.cifar100, cnfg.files_list_cifar100 , cnfg.label_head_cifar100)
    cifar100, image_data = choose_x_labels_from_cifar100(cifar100, image_data, np.array(categorys_numbers), old_csv)
    write_df_to_csv(cifar100)
    #save_images(cifar100, image_data)
    return cifar100


def split_train_test_validation():
    # read data from csv and split cifar10 and cifar100
    cifar10_100 = pd.read_csv(cnfg.csv_path)
    cifar10 = cifar10_100[cifar10_100['source_image'] == cnfg.cifar10]
    cifar100 = cifar10_100[cifar10_100['source_image'] ==cnfg.cifar100]
    cifar100 = shuffle(cifar100)
    personal=cifar10_100[cifar10_100['source_image'] == cnfg.personal]
    len_cifar_10 = len(cifar10)
    len_cifar_100 = len(cifar100)
    # split cifar10 to train validation test
    x_train = cifar10.iloc[:int(cnfg.train_part * len_cifar_10), :-1]
    y_train = cifar10.iloc[:int(cnfg.train_part * len_cifar_10), -1]
    x_validation = cifar10.iloc[int(cnfg.train_part * len_cifar_10):int((cnfg.train_part+cnfg.validation_part) * len_cifar_10), :-1]
    y_validation = cifar10.iloc[int(cnfg.train_part * len_cifar_10):int((cnfg.train_part+cnfg.validation_part) * len_cifar_10), -1]
    x_test = cifar10.iloc[int((1-cnfg.test_part) * len_cifar_10):, :-1]
    y_test = cifar10.iloc[int((1-cnfg.test_part) * len_cifar_10):, -1]
    # split cifar100 to train validation test
    # x_train = pd.concat([x_train, cifar100[cifar100['batch'] == 'train'].iloc[:int(cnfg.train_part * len_cifar_100), :-1]])???
    x_train = pd.concat([x_train, cifar100.iloc[:int(cnfg.train_part * len_cifar_100), :-1]])
    y_train = pd.concat([y_train, cifar100.iloc[:int(cnfg.train_part * len_cifar_100), -1]])
    x_validation = pd.concat([x_validation, cifar100.iloc[int(cnfg.train_part * len_cifar_100):int((cnfg.train_part+cnfg.validation_part)  * len_cifar_100), :-1]])
    y_validation = pd.concat([y_validation, cifar100.iloc[int(cnfg.train_part * len_cifar_100):int((cnfg.train_part+cnfg.validation_part)  * len_cifar_100), -1]])
    x_test = pd.concat([x_test, cifar100.iloc[int((1-cnfg.test_part) * len_cifar_100):, :-1]])
    y_test = pd.concat([y_test, cifar100.iloc[int((1-cnfg.test_part) * len_cifar_100):, -1]])
    #add personal images to test
    x_test = pd.concat([x_test, personal[:-1]])
    y_test = pd.concat([y_test, personal.iloc[:,-1]])

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def save_images_as_nparray():
    Xtrain, Xvalidation, Xtest, Ytrain, Yvalidation, Ytest = split_train_test_validation()

    images_Xtrain = [cv2.imread(Xtrain.iloc[i]['image_path'] + Xtrain.iloc[i]["image_name"]) for i in range(len(Xtrain))]
    images_Xvalidation = [cv2.imread(Xvalidation.iloc[i]['image_path'] + Xvalidation.iloc[i]["image_name"]) for i in range(len(Xvalidation))]
    images_Xtest = [cv2.imread(Xtest.iloc[i]['image_path'] + Xtest.iloc[i]["image_name"]) for i in range(len(Xtest))]

    images_Xtrain = np.array((images_Xtrain))
    images_Xvalidation = np.array((images_Xvalidation))
    images_Xtest = np.array((images_Xtest))
    Ytrain = np.array((Ytrain))
    Yvalidation = np.array((Yvalidation))
    Ytest = np.array((Ytest))
    return images_Xtrain, images_Xvalidation, images_Xtest, Ytrain, Yvalidation, Ytest


def save_to_zip():
    images_Xtrain, images_Xvalidation, images_Xtest, Ytrain, Yvalidation, Ytest = save_images_as_nparray()
    np.savez(cnfg.z_file_path, train=images_Xtrain, ytrain=Ytrain, validation=images_Xvalidation,
             yvalidation=Yvalidation, test=images_Xtest, ytest=Ytest)
