import cv2
import pandas as pd
import keras
import numpy as np
from PIL import Image
from keras.utils import load_img, img_to_array, save_img

import config as cnfg
import create_data as cd


def load_compiled_model():
    model = keras.models.load_model(cnfg.model_path)
    return model


def load_history():
    history=pd.read_csv(cnfg.history_path)
    return history


def load_data():
    loaded_data = np.load('./'+cnfg.z_file_path)
    x_train = loaded_data['train'].astype('float32')/255
    x_validation = loaded_data['validation'].astype('float32')/255
    x_test = loaded_data['test'].astype('float32')/255
    y_train = loaded_data['ytrain']
    y_test = loaded_data['ytest']
    y_validation = loaded_data['yvalidation']
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def load_our_labels():
    return cd.get_model_classes_dict()


# load image with keras
def preprecessing_for_predict1(image):
    if isinstance(image, str):
        image = load_img(image, target_size=(32, 32))
    image=img_to_array(image)
    image = image.reshape(-1, 32, 32, 3)
    image = image.astype('float32')
    image /= 255
    return image


def predict_by_image1(image):
    model = load_compiled_model()
    image=preprecessing_for_predict(image)
    prediction = model.predict(image,verbose=0)
    pred = np.argsort(prediction)
    pred = pred[0][-3:]
    labels = [cd.get_model_classes_dict()[pred[-1]], cd.get_model_classes_dict()[pred[-2]],
              cd.get_model_classes_dict()[pred[-3]]]
    percent = ["%5.2f" % (float(prediction[0][pred[-1]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-2]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-3]]) * 100) + "%"]
    res_dict= {labels[i]: percent[i] for i in range(len(percent))}
    return res_dict


#load image with cv2
def preprecessing_for_predict(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = image.reshape(-1, 32, 32, 3)
    image = image.astype('float32')
    image /= 255
    return image


def predict_by_image(image):
    model = load_compiled_model()
    image = preprecessing_for_predict(image)
    prediction = model.predict(image,verbose=0)
    pred = np.argsort(prediction)
    pred = pred[0][-3:]
    labels = [cd.get_model_classes_dict()[pred[-1]], cd.get_model_classes_dict()[pred[-2]],
              cd.get_model_classes_dict()[pred[-3]]]
    percent = ["%5.2f" % (float(prediction[0][pred[-1]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-2]]) * 100) + "%",
               "%5.2f" % (float(prediction[0][pred[-3]]) * 100) + "%"]
    res_dict= {labels[i]: percent[i] for i in range(len(percent))}
    return res_dict


#
# def wrong_preprocessing(image):
#
#     image = Image.open(image)
#     print(Image)
#     image.save("outt.png")
#     imagecv2=cv2.imread("outt.png")
#     image=cv2.resize(imagecv2, (32, 32), interpolation=cv2.INTER_AREA)
#     # print(imagecv2.shape)
#
#     # image = np.resize(imagecv2, (32, 32, 3))
#     print(image.size)
#     print(image.shape)
#     print(image)
#     cv2.imwrite("out.png", image)
#     image = image.reshape(-1, 32, 32, 3)
#     print(image.shape)
#
#     image = image.astype('float32')
#     image /= 255
#
# wrong_preprocessing('output0.png')
# preprecessing_for_predict1('output0.png')