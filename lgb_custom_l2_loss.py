from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')

import lightgbm as lgb

from sklearn.metrics import mean_squared_error, accuracy_score
#%%
dataset = datasets.load_boston()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
#%%
def l2_loss(y, data):
    t = data.get_label()
    w  =data.get_weight()
    # print(w)
    grad = y - t
    hess = np.ones_like(y)
    return grad, hess

def l2_eval(y, data):
    t = data.get_label()
    loss = (y - t) ** 2
    return 'l2', loss.mean(), False
#%%
lgb_train = lgb.Dataset(X_train, y_train)

# Using built-in objective
lgbm_params = {
    'objective': 'regression',
    'random_seed': 0,
    "boost_from_average": False,
    "lambda_l2": 0.5
    }
model = lgb.train(lgbm_params,
                  lgb_train)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
# 10.74183797847384

# Using custom objective
lgbm_params = {
    'random_seed': 0,
    "boost_from_average": False,
    "lambda_l2": 0.5
    }

model = lgb.train(lgbm_params,
                  lgb_train,
                  fobj=l2_loss,
                  feval=l2_eval)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
# 10.74183797847384
