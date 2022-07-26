import pandas as pd
import keras
import numpy as np
from PIL import Image

import config as cnfg
import create_data as cd

def load_compiled_model():
    model = keras.models.load_model(cnfg.model_path)
    # model = keras.models.load_model("C://Users//1//Downloads//keras_cifar10_trained_model_1A.h5")
    return model

def load_history():
    history=pd.read_csv(cnfg.history_path)
    print(history.head())
    return history

def load_data():
    loaded_data = np.load('./cfar10_modified_1000.npz')
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
    return  x_train, x_validation, x_test, y_train, y_validation, y_test

def load_our_labels():
    DATA = pd.read_csv(cnfg.csv_path)
    value_count_dict = dict(DATA['label'].value_counts())
    value_count_dict = {k: value_count_dict[k] for k in sorted(value_count_dict)}
    labels_array = {label:cd.create_classes_dict()[label] for label in value_count_dict.keys()}
    return labels_array

def predict_by_image(image):
    print(type(image))
    model = load_compiled_model()
    print("p nodel")

    if isinstance(image, str):
        image = Image.open(image)
    print("aaa")
    image = np.resize(image,(32, 32,3))
    image = image.reshape(-1, 32, 32,3)
    image = image.astype('float32')
    image/=255
    # pred = np.array(model.predict(image)[0])
    pred = model.predict(image)[0]
    arr = [1, 2, 3, 4, 5, 6, 7]
    arr = arr[0:0] + arr[7:]
    print(arr)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    print(pred)
    print(np.sum(pred))
    print(pred.argmax())
    print(load_our_labels().values())
    argmax1=pred.argmax()
    print(argmax1)
    max1=pred[argmax1]
    print(max1)
    pred=pred[1:2]+pred[argmax1+1:]
    print(np.sum(pred))
    argmax2= pred.argmax()
    max2 = pred[argmax2]
    print("67")
    labels=load_our_labels()
    print(labels)
    print(labels.values()[0])
    # res_dict={labels[argmax1]:max1,labels[argmax2]:max2}//אם זה לפי המקומות ולא הקי בדיקט
    res_dict={labels.values()[argmax1]:max1,labels.values()[argmax2]:max2}
    print(res_dict)
    return res_dict
