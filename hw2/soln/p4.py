"""
Start file for hw1pr1 of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

First, fill in the the code of step 0 in the main driver to load the data, then
please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
	1) Let m be the number of samples
	2) Let n be the number of features

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################
#			 Helper Functions	    	#
#########################################
def predict(W, X):
	"""	This function takes in two arguments:
			1) W, a weight matrix with bias
			2) X, the data with dimension m x (n + 1)

		This function calculates and returns the predicted label, y_pred.

		NOTE: You don't need to change this function.
	"""
	return X @ W

def find_cost(X, y, W , reg):
	"""	This function takes in three arguments:
			1) W, a weight matrix with bias
			2) X, the data with dimension m x (n + 1)
			3) y, the label of the data with dimension m x 1

		This function calculates and returns the l1 regularized 
		mean-squared error
	"""
	# TODO: Solve for l1-regularized mse
	"*** YOUR CODE HERE ***"
	err = X @ W - y
	err = float(err.T @ err)
	return (err + reg * np.abs(W).sum())/len(y)
	"*** END YOUR CODE HERE ***"

def find_grad(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix with bias
			4) reg, the parameter for regularization

		This function calculates and returns the gradient of W
	"""
	"*** YOUR CODE HERE ***"
	m = X.shape[0]
	return X.T @ (X @ W - y) / m
	"*** END YOUR CODE HERE ***"

def prox(X, gamma):
	""" This function takes in two arguments:
			1)  X, a vector
			2) gamma, a scalar
		
		This function thresholds each entry of X with gamma
		and updates the changes in place.
	"""
	"""*** YOUR CODE HERE ***"""
	X[np.abs(X) <= gamma] = 0.
	X[X > gamma] -= gamma
	X[X < -gamma] += gamma
	""" END YOUR CODE HERE """
	return X

def grad_lasso(
	X, y, reg=1e-6, lr=1e-3, eps=1e-5,
	max_iter=300, batch_size=256, print_freq=1):
	""" This function takes in the following arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) reg, the parameter for regularization
			4) lr, the learning rate
			5) eps, the threshold of the norm for the gradients
			6) max_iter, the maximum number of iterations
			7) batch_size, the size of each batch for gradient descent
			8) print_freq, the frequency of printing the report
		
		This function returns W, the optimal weight, 
		by lasso gradient descent.
	"""
	m, n = X.shape
	obj_list = []
	# initialize the weight and its gradient
	W = np.linalg.solve(X.T @ X, X.T @ y)
	W_grad = np.ones_like(W)
	print('==> Running gradient descent...')
	iter_num = 0
	t_start = time.time()
	"*** YOUR CODE HERE ***"
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# update W
		ind = np.random.randint(0, m, size=batch_size)
		W_grad = find_grad(X[ind], y[ind], W, reg=reg)
		# apply threshold function
		W = prox(W - lr * W_grad, reg * lr)
		# calculate MSE
		cost = find_cost(X[ind], y[ind], W, reg=reg)
		obj_list.append(cost)
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration{} - training cost {: .4E} - \
				sparsity {: .2f}'.format(iter_num + 1, cost, \
					(np.abs(W) < reg * lr).mean()))
		iter_num += 1
	"*** END YOUR CODE HERE ***"
	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} \
		seconds'.format(t = t_end - t_start))
	return W

###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')

	# Read data
	df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', \
		sep=', ', engine='python')
	X = df[[col for col in df.columns if col not in ['url', 'shares', 'cohort']]]
	y = np.log(df.shares).values.reshape(-1,1)
	X = np.hstack((np.ones_like(y), X))
	# =============STEP 1: LASSO GRADIENT DESCENT=================
	# NOTE: Fill in code in find_MSE, find_grad, prox and 
	# grad_lasso for this step
	W = grad_lasso(X, y, reg=1e10)
