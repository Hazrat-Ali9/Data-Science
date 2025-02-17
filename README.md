# Hazrat Ali

import pandas as pd
import numpy as np

from itertools import cycle

import sklearn
from sklearn.model_selection import train_test_split
#cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#model validation
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,accuracy_score,
    roc_auc_score,roc_curve,auc,
    classification_report,mean_absolute_error,
    mean_squared_error,cohen_kappa_score,
    log_loss,precision_score,f1_score,recall_score,fbeta_score,matthews_corrcoef)

#machine learning algorithms
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
#from catboost import CatBoostClassifier,Pool,cv,CatBoostRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.linear_model as lm
import xgboost as xgb
from xgboost import plot_tree,plot_importance
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow import keras
from scipy import stats # Removed interp
from scipy.interpolate import interp1d # import interp1d from scipy.interpolate
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import tree

#data visualization
import plotly.express as px
%matplotlib inline
import matplotlib.pyplot as plt
#import matplotlib as mpl
#import matplotlib.pylab as pylab
import seaborn as sns
#mpl.style.use('ggplot')
sns.set_style('white')
#pylab.rcParams['figure.figsize']=10,8

import warnings
warnings.filterwarnings('ignore')

# from matplotlib import style
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from matplotlib.gridspec import GridSpec
from plotly.subplots import make_subplots
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)
# sns.set()
# #style.use('fivethirtyeight')
# pd.options.mode.chained_assignment = None


# Data Science Math Command : 
 
    df = pd.read_csv('dataset.csv')
    df.head()
    df.tail()
    df.info()
    df.describe()


    

