import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(y_test, y_pred, classes, title='',
                          normalize=False,
                          cmap='Blues',
                          linecolor='k'):
    
    cm = np.array(confusion_matrix(y_test, y_pred))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_title = 'Confusion matrix, with normalization'
    else:
        cm_title = title

    fmt = '.3f' if normalize else 'd'
    sns.heatmap(cm, fmt=fmt, annot=True, square=True,
                xticklabels=classes, yticklabels=classes,
                cmap=cmap,
                linewidths=0.5, linecolor=linecolor,
                cbar=False)
    sns.despine(left=False, right=False, top=False, bottom=False)

    plt.title(cm_title)
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()



def pretty_classification_report(y_test, y_pred, labels=None, cmap='viridis'):
    df = pd.DataFrame(classification_report(y_pred, 
                                            y_test, digits=2, 
                                            target_names=labels,
                                            output_dict=True)).T
    df['support'] = df.support.apply(int)

    return  df.style.background_gradient(cmap=cmap,
                                subset=pd.IndexSlice['0':'9', :'f1-score'])
