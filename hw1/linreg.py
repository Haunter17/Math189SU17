import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================================================
# For the following sections:
# Let m be the number of samples
# Let n be the number of features
# ==============HELPER FUNCTIONS============
def linreg(X, y, reg=0.0):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.
		reg is the parameter for regularization.

		Return the optimal weight matrix.
	'''
	# Hint: Find the numerical solution for part c
	# Use np.eye to create identity matrix
	# Use np.linalg.solve to solve W_opt

	# YOUR CODE GOES BELOW
	eye = np.eye(X.shape[1])
	eye[0, 0] = 0 # don't regularize the bias term
	W_opt = np.linalg.solve(X.T @ X + reg * eye, X.T @ y)
	return W_opt

def predict(W, X):
	'''
		W is a weight matrix with bias.
		X is the data with dimension m x (n + 1).

		Return the predicted label, y_pred.
	'''
	return X * W

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
	m = X.shape[0]
	MSE = np.linalg.norm(diff, 2) / m
	return np.sqrt(MSE)

def RMSE_vs_lambda(X_train, y_train, X_val, y_val):
	'''
		X is the data with dimension m x (n + 1).
		y is the label with dimension m x 1.

		Genearte a plot of RMSE vs lambda.
		Return the regularization parameter that minimizes RMSE.
	'''
	# Set up plot style
	plt.style.use('ggplot')

	RMSE_list = []
	# Construct a list of regularization parameters with random uniform sampling
	# Then, generate a list of W_opt's according to these parameters
	# Finally, generate a list of RMSE according to reg_list
	# YOUR CODE GOES BELOW
	reg_list = np.random.uniform(0.0, 150.0, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = lb) for lb in reg_list]
	for index in range(len(reg_list)):
		W_opt = W_list[index]
		RMSE_list.append(find_RMSE(W_opt, X_val, y_val))

	# Plot RMSE vs lambda
	RMSE_vs_lambda_plot, = plt.plot(reg_list, RMSE_list)
	plt.setp(RMSE_vs_lambda_plot, color = 'red')
	plt.title('RMSE vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('RMSE')
	plt.savefig('RMSE_vs_lambda.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	# Find the regularization value that minimizes RMSE
	# YOUR CODE GOES BELOW
	opt_lambda_index = np.argmin(RMSE_list)
	reg_opt = reg_list[opt_lambda_index]
	return reg_opt

def norm_vs_lambda(X_train, y_train, X_val, y_val):
	'''
		X is the data with dimension m x (n + 1).
		y is the label with dimension m x 1.

		Genearte a plot of norm of the weights vs lambda.
	'''
	# You may reuse the code for RMSE_vs_lambda
	# to generate the list of weights and regularization parameters
	# YOUR CODE GOES BELOW
	reg_list = np.random.uniform(0.0, 150.0, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = lb) for lb in reg_list]

	# Calculate the norm of each weight
	# YOUR CODE GOES BELOW
	norm_list = [np.linalg.norm(W, 2) for W in W_list]

	# Plot norm vs lambda
	norm_vs_lambda_plot, = plt.plot(reg_list, norm_list)
	plt.setp(norm_vs_lambda_plot, color = 'blue')
	plt.title('norm vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('norm')
	plt.savefig('norm_vs_lambda.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

def linreg_no_bias(X, y, reg = 0.0):
	'''
		X is matrix with dimension m x n.
		y is label with dimension m x 1.
		reg is the parameter for regularization.

		Return the optimal weight and bias separately.
	'''
	# Find the numerical solution in part d
	# YOUR CODE GOES BELOW
	m = X.shape[0]
	ones = np.eye(m)
	Aggregate = X.T @ (np.eye(m) - np.ones(m) / m)
	W_opt = np.linalg.solve(Aggregate @ X + reg * np.eye(Aggregate.shape[0]), \
		Aggregate @ y)
	b_opt = sum((y - X @ W_opt)) / m
	return b_opt, W_opt

# *****************************************************************
# ====================main driver function=========================
if __name__ == '__main__':
	# Part c
	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')
	train_pct = 2.0 / 3
	val_pct = 5.0 / 6
	df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', \
		sep = ', ', engine = 'python')
	# split the data frame by type: training, validation, and test
	df['type'] = ''
	df.loc[:int(train_pct * len(df)), 'type'] = 'train'
	df.loc[int(train_pct * len(df)) : int(val_pct * len(df)), 'type'] = 'val'
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
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_val = np.hstack((np.ones_like(y_val), X_val))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# Convert data to matrix
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_val = np.matrix(X_val)
	y_val = np.matrix(y_val)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)

	# =============STEP 1: RMSE vs lambda=================
	print('==> Step 1: RMSE vs lambda...')
	# Fill in the code in linreg, findRMSE, and RMSE_vs_lambda
	reg_opt = RMSE_vs_lambda(X_train, y_train, X_val, y_val)
	print('==> The optimal regularization parameter is {reg: 4.4f}.'.format(\
		reg = reg_opt))
	# Find the optimal weights and bias for future use in step 3
	W_with_b_1 = linreg(X_train, y_train, reg = reg_opt)
	b_opt_1 = W_with_b_1[0]
	W_opt_1 = W_with_b_1[1: ]

	# =============STEP 2: Norm vs lambda=================
	print('==> Step 2: RMSE vs lambda...')
	# Fill in the code in norm_vs_lambda
	norm_vs_lambda(X_train, y_train, X_val, y_val)

	# Part d
	# =============STEP 3: Linear regression without bias=================
	# From here on, we will strip the columns of ones for all data
	X_train = X_train[:, 1:]
	X_val = X_val[:, 1:]
	X_test = X_test[:, 1:]
	# Fill in the code in linreg_no_bias
	# Compare the result with the one from step 1
	# The difference in norm should be a small scalar (i.e, 1e-10)
	print('==> Step 3: Linear regression without bias...')
	b_opt_2, W_opt_2 = linreg_no_bias(X_train, y_train, reg = reg_opt)
	diff_bias = np.linalg.norm(b_opt_2 - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff = diff_bias))
	diff_W = np.linalg.norm(W_opt_2 -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff = diff_W))
