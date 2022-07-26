import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import compiled_model as c_model

#intialize
labels = c_model.load_our_labels()
x_train, x_validation, x_test, y_train, y_validation, y_test = c_model.load_data()
model = c_model.load_compiled_model()
y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels,rotation='vertical')
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return im, cbar

def annotate_heatmap(im, data=None, fmt="d", threshold=None):
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment="center",
                                color="white" if data[i, j] > threshold else "black")
            texts.append(text)
    return texts

def show_confusion_matrix():
    cm = confusion_matrix(Y_true, Y_pred_classes)
    thresh = cm.max() / 2.
    fig, ax = plt.subplots(figsize=(12, 12))
    im, cbar = heatmap(cm, labels.values(), labels.values(), ax=ax,
                       cmap=plt.cm.Blues, cbarlabel="count of predictions")
    texts = annotate_heatmap(im, data=cm, threshold=thresh)

    fig.tight_layout()
    plt.show()

def plotmodelhistory():
    history=c_model.load_history()
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(history['accuracy'])
    axs[0].plot(history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()

def show_samples_after_predict():
    R = 5
    C = 5
    fig, axes = plt.subplots(R, C, figsize=(12, 12))
    axes = axes.ravel()
    for i in np.arange(0, R * C):
        axes[i].imshow(x_test[i])
        axes[i].set_title("True: %s \nPredict: %s" % (labels[Y_true[i]], labels[Y_pred_classes[i]]))
        axes[i].axis('off')
        plt.subplots_adjust(wspace=1)
    plt.show()

def show_wrong_predicts():
    R = 3
    C = 5
    fig, axes = plt.subplots(R, C, figsize=(12, 8))
    axes = axes.ravel()
    misclassified_idx = np.where(Y_pred_classes != Y_true)[0]
    for i in np.arange(0, R * C):
        axes[i].imshow(x_test[misclassified_idx[i]])
        axes[i].set_title("True: %s \nPredicted: %s" % (labels[Y_true[misclassified_idx[i]]],
                                                        labels[Y_pred_classes[misclassified_idx[i]]]))
        axes[i].axis('off')
        plt.subplots_adjust(wspace=1)
    plt.show()

