import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
def roc_points(a,b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]
    fpr,tpr,_=roc_curve(y,np.r_[a,b]); return fpr,tpr
def auroc(a,b):
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; return float(roc_auc_score(y,np.r_[a,b]))
