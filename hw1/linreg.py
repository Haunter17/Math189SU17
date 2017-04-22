import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============STEP 0: LOADING DATA=================
print('Loading data...')
train_pct = 2.0 / 3
val_pct = 5.0 / 6
df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', \
	sep = ', ', engine = 'python')
# split the data frame by type: training, validation, and test
df['type'] = ''
df.loc[:int(train_pct * len(df)), 'type'] = 'train'
df.loc[int(train_pct * len(df)) * int(val_pct * len(df)), 'type'] = 'val'
df.loc[int(val_pct * len(df)):, 'type'] = 'test'
# extracting columns into training, validation, and test data
X_train = np.array(df[df.type == 'train'][[col for col in df.columns \
	if col not in ['url', 'shares', 'type']]])
y_train = np.log(df[df.type == 'train'].shares).reshape((-1, 1))
X_val = np.array(df[df.type == 'val'][[col for col in df.columns \
	if col not in ['url', 'shares', 'type']]])
y_val = np.log(df[df.type == 'val'].shares).reshape((-1, 1))
X_test = np.array(df[df.type == 'test'][[col for col in df.columns \
	if col not in ['url', 'shares', 'type']]])
y_test = np.log(df[df.type == 'test'].shares).reshape((-1, 1))

# Stack a column of ones to the feature data
# Use np.ones / np.ones_like to create a column of ones
# Use np.hstack to stack the column to the matrix

# YOUR CODE GOES BELOW
X_train = np.hstack((np.ones(y_train.shape), X_train))
X_val = np.hstack((np.ones(y_val.shape), X_val))
X_test = np.hstack((np.ones(y_test.shape), X_test))

# Convert data to matrix
X_train = np.matrix(X_train)
y_train = np.matrix(y_train)
X_val = np.matrix(X_val)
y_val = np.matrix(y_val)
X_test = np.matrix(X_test)
y_test = np.matrix(y_test)

# ===================================================
# For the following sections:
# Let m be the number of samples
# Let n be the number of features
# ==============STEP 1: Linear Regression============
def linreg(X, y, reg=0.0):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.
		reg is the parameter for regularization.

		Return the optimal weight matrix.
	'''
	# Hint: W_opt = (X.T * X + reg * I)^(-1) * X.T * y
	# Use np.eye to create identity matrix
	# Use np.linalg.solve to solve W_opt

	# YOUR CODE GOES BELOW
	eye = np.eye(X.shape[1])
	eye[0, 0] = 0 # don't regularize the bias term
	return np.linalg.solve(X.transpose() * X + reg * eye, X.transpose() * y)

def predict(W, X):
	'''
		W is a weight matrix with bias.
		X is the data with dimension m x (n + 1).

		Return the predicted label, y_pred.
	'''
	return W.T * X

def find_RMSE(W, X, y):
	'''
		W is the weight matrix with bias.
		X is the data with dimension m x (n + 1).
		y is label with dimension m x 1.

		Return the root mean-squared error.
	'''
	# YOUR CODE GOES BELOW
	y_pred = predict(W, X)
	diff = y - y_pred
	m = y.shape[0]
	MSE = np.linalg.norm(diff, 2) / m
	return np.sqrt(MSE)

def RMSE_vs_lambda(X, y, reg_list):
	'''
		W is the weight matrix with bias.
		X is the data with dimension m x (n + 1).
		reg_list is a list of regularization parameters.

		Genearte a few plots as described below.
	'''
	# Set up plot style
	plt.style.use('ggplot')

