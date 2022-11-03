
import numpy as np
import os
import config as cnfg

import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.utils import load_img, img_to_array
# from classification_models.tfkeras import Classifiers
# import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

from pdb import set_trace as bp


def process_sample(img_path, dest_dir, model, explainer):
    img = load_img(img_path, target_size=(32, 32))
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    # bp()
    x = img_to_array(img)
    image = x.reshape(-1, 32, 32, 3)
    image = image.astype('float32')
    image /= 255
    preds = model.predict(image, verbose=0)[0]
    label = np.argmax(preds)
    explanation = explainer.explain_instance(x.astype('double'), model.predict, top_labels=2, hide_color=0,
                                             num_samples=1)  # TODO: batch?
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5,
                                                hide_rest=False)
    img_contour = mark_boundaries(temp, mask)
    img_contour = img_contour / 255.0
    plt.imsave(dest_dir + '/' + base_name + '_lime.png', img_contour)
    return preds[label]


def main():
    model_path = cnfg.model_path
    dataset_path = cnfg.csv_path
    dest_dir = cnfg.dest_dir

    if not os.path.exists(model_path):
        print('Model {} doesn\'t exist'.format(model_path))
    if not os.path.exists(dataset_path):
        print('Can\'t find dataset at {}'.format(model_path))
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    sample = 'output0.png'
    results_dict = {'path': [], 'score': []}
    model = keras.models.load_model(model_path)
    explainer = lime_image.LimeImageExplainer()
    score = process_sample(sample, dest_dir, model, explainer)
    results_dict['path'].append(sample)
    results_dict['score'].append(score)
    df = pd.DataFrame(results_dict)
    df.describe()
    df = df.sort_values(by='score', ascending=False)
    df.to_csv(dest_dir + '/scores.csv')


main()