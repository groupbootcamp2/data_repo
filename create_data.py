import numpy as np
import pandas as pd
import os
from PIL import Image as im
from typing import List, Dict
from sklearn.utils import shuffle
import cv2
from PIL import Image
import config as cnfg


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_cifar10_classes_dict():
    meta_dict10 = unpickle(f"..\\{cnfg.cifar10}\\{cnfg.meta_file_cifar10}")
    classes_dict = {np.arange(0, cnfg.num_classes_cifar10)[i]: (str(meta_dict10[b'label_names'][i]))[2:-1] for i in range(cnfg.num_classes_cifar10)}
    return classes_dict

def create_cifar100_classes_dict():
    meta_dict100 = unpickle(f"..\\{cnfg.cifar100}\\{cnfg.meta_file_cifar100}")
    classes_dict100 = {np.arange(0,cnfg.num_classes_cifar100)[i]: (str(meta_dict100[b'coarse_label_names'][i]))[2:-1] for i in range(cnfg.num_classes_cifar100)}
    return classes_dict100



def create_df(dataset_dirname: str, files_list: List[str], labels: str):
    df = pd.DataFrame()
    image_data = [[0] * 3072]
    for i in range(len(files_list)):
        file = os.path.join(cnfg.dataset_dir_name,dataset_dirname, files_list[i])
        dict = unpickle(file)
        dict_len = len(dict[b'filenames'])
        new_dict = {"image_name": dict[b'filenames'], "image_path": [cnfg.images_dir_path] * dict_len,
                   "source_image": [dataset_dirname] * dict_len, "batch": [files_list[i]] * dict_len, "label": dict[labels]}
        new_dict["image_name"] = [(str(sub)).replace("'", "")[1:] for sub in new_dict["image_name"]]
        newDF = pd.DataFrame(new_dict)
        df = pd.concat([df, newDF])
        image_data.extend(dict[b'data'])
    return df, image_data[1:]

def save_image( data_arr, path_to_save):
    array = np.array(data_arr)
    vector_size=cnfg.image_size*cnfg.image_size #vector_size_for_each_color
    rgb_array = np.zeros((cnfg.image_size, cnfg.image_size, 3), 'uint8')
    rgb_array[..., 0] = array[:vector_size].reshape(cnfg.image_size, cnfg.image_size)
    rgb_array[..., 1] = array[vector_size:vector_size*2].reshape(cnfg.image_size, cnfg.image_size)
    rgb_array[..., 2] = array[vector_size*2:vector_size*3].reshape(cnfg.image_size, cnfg.image_size)
    img = im.fromarray(rgb_array)
    img.save(path_to_save)


def save_images(df, data):
    for arr in range(len(data)):
        save_image(data[arr],df.iloc[arr]["image_path"] + df.iloc[arr]["image_name"])


def choose_x_labels_from_cifar100(df, imgs, cifar100_source_new_labels:Dict, old_csv):
    images = [[0] * cnfg.image_size*cnfg.image_size*3]
    df['index'] = [i for i in range(len(df['label']))]
    df = df[(df['label'].isin(cifar100_source_new_labels.keys()))]
    df['label'] = [cifar100_source_new_labels[label] for label in df["label"]]
    df = df[(~df['label'].isin(old_csv['label'].value_counts().keys()))]
    for i in df['index']:
        images.append(imgs[i])
    df.drop('index', inplace=True, axis=1)
    return df, images[1:]




def insert_personal_image_to_csv (image_name:str, label:int ):
    df = pd.DataFrame({"image_name": [image_name], "image_path": [cnfg.images_dir_path], "source_image":['personal'], "batch": ['test'], "label": [label]})
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
    save_images(df, image_data)
    save_cifar10_dict_labels_to_zip()
    return df


def load_x_labels_from_cifar100(categorys_numbers: List[int]):
    cifar100_source_labels=update_cifar100_dict_labels_to_zip(categorys_numbers)
    old_csv = pd.DataFrame(pd.read_csv(cnfg.csv_path))
    cifar100, image_data = create_df(cnfg.cifar100, cnfg.files_list_cifar100 , cnfg.label_head_cifar100)
    cifar100, image_data = choose_x_labels_from_cifar100(cifar100, image_data, cifar100_source_labels, old_csv)
    write_df_to_csv(cifar100)
    save_images(cifar100, image_data)

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
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_train_test_validation()

    images_xtrain = [cv2.imread(x_train.iloc[i]['image_path'] + x_train.iloc[i]["image_name"]) for i in range(len(x_train))]
    images_xvalidation = [cv2.imread(x_validation.iloc[i]['image_path'] + x_validation.iloc[i]["image_name"]) for i in range(len(x_validation))]
    images_xtest = [cv2.imread(x_test.iloc[i]['image_path'] + x_test.iloc[i]["image_name"]) for i in range(len(x_test))]

    images_xtrain = np.array(images_xtrain)
    images_xvalidation = np.array(images_xvalidation)
    images_xtest = np.array(images_xtest)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)
    y_test = np.array(y_test)
    return images_xtrain, images_xvalidation, images_xtest, y_train, y_validation, y_test
save_images_as_nparray()

def save_to_zip():
    images_xtrain, images_xvalidation, images_xtest, y_train, y_validation, y_test = save_images_as_nparray()
    np.savez(cnfg.z_file_path, train=images_xtrain, ytrain=y_train, validation=images_xvalidation,
             yvalidation=y_validation, test=images_xtest, ytest=y_test)

def save_cifar10_dict_labels_to_zip():
    labels_dict= create_cifar10_classes_dict()
    np.savez(cnfg.z_label_dict_path, labels_dict=labels_dict)

def update_cifar100_dict_labels_to_zip(labels_numbers_to_update:List[int]) :
    cifar100_labels_dict = create_cifar100_classes_dict()
    our_labels_names = [cifar100_labels_dict[l] for l in labels_numbers_to_update]
    labels_dict = get_model_classes_dict()
    max_class=list(labels_dict.keys())[-1]
    new_labels_number=np.arange(max_class+1,max_class+1+len(labels_numbers_to_update))
    cifar100_source_new_labels={labels_numbers_to_update[i]:new_labels_number[i] for i in range(len(labels_numbers_to_update))}
    for i in range(len(labels_numbers_to_update)):
        max_class += 1
        labels_dict.update({max_class:our_labels_names[i]})
    np.savez(cnfg.z_label_dict_path, labels_dict=labels_dict)
    return cifar100_source_new_labels

def get_model_classes_dict():
    return (np.load('./' + cnfg.z_label_dict_path ,allow_pickle=True)['labels_dict']).tolist()



