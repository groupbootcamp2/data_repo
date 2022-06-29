import numpy as np
import pandas as pd
import os
from PIL import Image as im
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_df(datasetType):
    df = pd.DataFrame()
    dataset="cifar"+str(datasetType)
    dirname="..\\cifar-10-batches-py"
    for i in range(1,6):
        file=os.path.join(dirname,"data_batch_"+str(i))
        dict=unpickle(file)
        DictLen=len(dict[ b'filenames'])
        path="..\\resources\\"
        newDict={"image_name":dict[ b'filenames'],"image_path":[path]*DictLen, "label": dict[b'labels'] , "source_image": [dataset]*DictLen, "batch":[i]*DictLen}
        newDict["image_name"]=[(str(sub)).replace("'", "")[1:] for sub in newDict["image_name"]]
        print(type(str(newDict["image_name"][0])))
        newDF=pd.DataFrame(newDict)

        for arr in range(DictLen):
            array = np.array(dict[b'data'][arr])
            rgbArray = np.zeros((32, 32, 3), 'uint8')
            rgbArray[..., 0] = array[:1024].reshape(32, 32)
            rgbArray[..., 1] = array[1024:2048].reshape(32, 32)
            rgbArray[..., 2] = array[2048:3072].reshape(32, 32)
            img = im.fromarray(rgbArray)
            img.save(path+newDF.iloc[arr]["image_name"])

        df=pd.concat([df,newDF])
    return df


def write_df_to_csv(df):
    df.to_csv("..\\cifar10.csv")
df=create_df(10)
write_df_to_csv(df)

csv=pd.read_csv("..\\cifar10.csv")
print(csv)
