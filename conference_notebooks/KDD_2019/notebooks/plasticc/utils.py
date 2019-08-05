import pandas as pd

import cudf as gd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def scatter(x,y,values,xlabel='x',ylabel='y',title=None,xlim=None):
    """
    Builds a scatter plot specific to the LSST data format by plotting 
    the flux over time for different passbands for a single object.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = np.array([colors[i] for i in values])
    ps = []
    bs = []
    bands = ['passband_%s'%i for i in ['u', 'g', 'r', 'i', 'z','y']]
    for i in sorted(np.unique(values)):
        mask = values==i
        if len(x[mask]):
            p = plt.scatter(x[mask],y[mask],c=colors[mask])
            ps.append(p)
            bs.append(bands[i])
    plt.legend(ps,bs,scatterpoints=1)
    if title is not None:
        plt.title(title)
    plt.grid()
    if xlim is None:
        plt.xlim([np.min(x)-10,np.min(x)+1500])
    else:
        plt.xlim(xlim)
    plt.ylabel('y: %s'%ylabel)
    plt.xlabel('x: %s'%xlabel)
    
    
def groupby_aggs(df,aggs,col = "object_id"):
    """
    Given a Dataframe and a dict of {"field":"["agg", "agg"]"}, perform the 
    given aggregations using the given column as the groupby key. The original
    (non-aggregated) field is dropped from the dataframe. 
    """

    res = None
    for i,j in aggs.items():
        for k in j:
            tmp = df.groupby(col,as_index=False).agg({i:[k]})
            tmp.columns = [col,'%s_%s'%(k,i)]
            res = tmp if res is None else  res.merge(tmp,on=[col],how='left')
        df.drop_column(i)
    return res


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(28, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
    return ax


def minimize_test_data(path, gpu_memory = 16):
    """
    This function shrinks the test data set based on GPU memory size,
    since the file is so large. 
    """
    PATH = path
    GPU_MEMORY = gpu_memory # GB.
    TEST_ROWS = 453653104 # number of rows in test data
    # no skip if your gpu has 32 GB memory
    # otherwise, skip rows porportionally
    OVERHEAD = 1.2 # cudf 0.7 introduces 20% memory overhead comparing to cudf 0.4
    SKIP_ROWS = int((1 - GPU_MEMORY/(32.0*OVERHEAD))*TEST_ROWS) 
    ts_cols = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected']
    test_gd = gd.read_csv('%s/test_set.csv'%PATH, names=ts_cols,skiprows=1+SKIP_ROWS)
    test_gd.to_csv("%s/test_set_minimal.csv"%PATH, index=False)

    
    
def cross_entropy(y_true, y_preds, classes):
    """
    Computes the weighted cross-entropy 
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true) # one-hot encodes y_true values
    
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)

    # Transform to log
    y_p_log = np.log(y_p)
    
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    
    class_weights = build_class_weights(classes)

    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    
    return loss


def xgb_multi_weighted_logloss(y_predicted, y_true, classes, class_weights):
    class_weights = build_class_weights(classes)
    
    loss = cross_entropy(y_true.get_label(), y_predicted, 
                                  classes)
    return 'wloss', loss

def build_class_weights(classes):
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    class_weights = {c: 1 for c in classes}
    class_weights.update({c:2 for c in [64, 15]})
    return class_weights


def xgb_cross_entropy_loss(classes):
    from functools import partial
    
    class_weights = build_class_weights(classes)

    return partial(xgb_multi_weighted_logloss, 
                        classes=classes, 
                        class_weights=class_weights)
