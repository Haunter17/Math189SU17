import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time

# *****************************************************************
# ====================helper functions=========================

def NLL(X, y, W, reg=0.0):
	'''
		Calculate negative log likelihood for softmax regression.
	'''
	# YOUR CODE GOES BELOW
	mu = X @ W # m x k
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	groundTruth = y * np.log(prob)
	return groundTruth.sum(axis=1).sum()

def grad_softmax(X, y, W, reg=0.0):
	'''
		Return the gradient of W for softmax regression.
	'''
	# YOUR CODE BELOW
	mu = X @ W
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	return X.T @ (prob - y) + reg * W

def predict(X, W):
	'''
		Return y_pred with dimension m x 1.
	'''
	mu = X @ W
	exp_mu = np.exp(mu)
	prob = exp_mu / exp_mu.sum(axis=1).reshape(-1, 1)
	y_pred = np.argmax(prob, axis=1).reshape(-1, 1)
	return y_pred

def get_accuracy(y_pred, y):
	'''
		Return the percentage of same bits between y_pred and y.
	'''
	diff = (y_pred == y).astype(int)
	accu = 1. * diff.sum() / len(y)
	return accu

def grad_descent(X, y, reg=0.0, lr=1e-5, eps=1e-6, \
	max_iter=500, print_freq=20):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.
		reg is the parameter for regularization.
		lr is the learning rate.
		eps is the threshold of the norm for the gradients.
		max_iter is the maximum number of iterations.
		print_freq is the frequency of printing the report.

		Return the optimal weight by gradient descent and 
		the corresponding learning objectives.
	'''
	m, n = X.shape
	k = y.shape[1]
	nll_list = []
	# initialize the weight and its gradient
	W = np.zeros((n, k))
	W_grad = np.ones((n, k))
	print('==> Running gradient descent...')
	iter_num = 0
	t_start = time.time()
	# Running the gradient descent algorithm
	# Update W
	# Calculate learning objectives
	# YOUR CODE GOES BELOW
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# calculate NLL
		nll = NLL(X, y, W, reg=reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)
		# calculate gradients and update W
		W_grad = grad_softmax(X, y, W, reg=reg)
		W -= lr * W_grad

		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - \
				negative log likelihood {: 4.4f}'.format(iter_num + 1, nll))
		iter_num += 1
	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} \
		seconds'.format(t=t_end - t_start))

	return W, nll_list

# *****************************************************************
# ====================main driver function=========================
if __name__ == '__main__':
	# =============STEP 0: LOADING DATA=================
	df_train = data.df_train
	df_test = data.df_test
	'''
		X is a matrix with dimension m x n
		y is a vector with dimension m x 1
	'''
	X_train = data.X_train
	y_train = data.y_train
	X_test = data.X_test
	y_test = data.y_test
	# stacking an array of ones
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_test = np.hstack((np.ones_like(y_test), X_test))
	# one hot encoder
	enc = OneHotEncoder()
	y_train_OH = enc.fit_transform(y_train.copy()).astype(int).toarray()
	y_test_OH = enc.fit_transform(y_test.copy()).astype(int).toarray()

	# =============STEP 1: Softmax Regression=================
	print('==> Step 1a: Running softmax regression with gradient descent...')
	# Fill in the code in NLL and grad_descent
	W_gd, nll_list_gd = grad_descent(X_train, y_train_OH, reg=1, lr=2e-5, \
		print_freq=25)
