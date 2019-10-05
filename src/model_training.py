import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

logging.basicConfig(format='%(levelname)s  %(asctime)s\t%(message)s', level=logging.INFO)

df = pd.read_pickle("../data/pruned.pkl")

train, test = train_test_split(df.select_dtypes(include=[np.number]), test_size = 0.25, random_state = 9)

xtrain = train[list(train.drop('Value', 1))]
ytrain = train[['Value']]

xtest = test[list(train.drop('Value', 1))]
ytest = test[['Value']]

# Linear Regression
lr = LinearRegression(fit_intercept=True)
lr.fit(xtrain, ytrain.values.ravel())

y_lr = lr.predict(xtest)

logging.info("Mean squared error: %.3f" % mean_squared_error(ytest, y_lr))
logging.info('R2 score: %.3f' % r2_score(ytest, y_lr))

cvScoreLR = cross_val_score(lr, xtest, ytest.values.ravel(), cv = 3, scoring = 'r2')
logging.info("Accuracy (cross validation score): %0.2f (+/- %0.2f)" % (cvScoreLR.mean(), cvScoreLR.std() * 2))

with open("./models/linreg.pkl", 'wb') as f:
    pickle.dump(lr, f)

# SVM Regressor
svr = SVR(kernel="linear")
svr.fit(xtrain, ytrain.values.ravel())

y_svr = svr.predict(xtest)

logging.info("Mean squared error: %.3f" % mean_squared_error(ytest, y_svr))
logging.info('R2 score: %.3f' % r2_score(ytest, y_svr))

cvScoreSVR = cross_val_score(svr, xtest, ytest.values.ravel(), cv = 3, scoring = 'r2')
logging.info("Accuracy (cross validation score): %0.2f (+/- %0.2f)" % (cvScoreSVR.mean(), cvScoreSVR.std() * 2))

with open("./models/svm_regressor.pkl", 'wb') as f:
    pickle.dump(svr, f)

# K-Nearest Neighbours
knn = neighbors.KNeighborsRegressor(n_neighbors = 10, weights = 'distance')
knn.fit(xtrain, ytrain)

y_knn = knn.predict(xtest)

logging.info("Mean squared error: %.3f" % mean_squared_error(ytest, y_knn))
logging.info('R2 score: %.3f' % r2_score(ytest, y_knn))

cvScoreKNN = cross_val_score(knn, xtest, ytest, cv = 5, scoring = 'r2')
logging.info("Accuracy (cross validation score): %0.2f (+/- %0.2f)" % (cvScoreKNN.mean(), cvScoreKNN.std() * 2))

with open("./models/knn.pkl", 'wb') as f:
    pickle.dump(knn, f)

# Random Forest Regressor
rf = RandomForestRegressor(random_state = 9, n_estimators = 100, criterion = 'mse')
rfPred = rf.fit(xtrain, ytrain.values.ravel())
y_rf = rfPred.predict(xtest)

logging.info("Mean squared error: %.3f" % mean_squared_error(ytest, y_rf))
logging.info('R2 score: %.3f' % r2_score(ytest, y_rf))

cvScoreRF = cross_val_score(rf, xtest, ytest.values.ravel(), cv = 3, scoring = 'r2')
logging.info("Accuracy (cross validation score): %0.2f (+/- %0.2f)" % (cvScoreRF.mean(), cvScoreRF.std() * 2))

with open("./models/random_forest.pkl", 'wb') as f:
    pickle.dump(rf, f)
