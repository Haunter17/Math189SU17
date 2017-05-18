"""
Start file for hw1pr2 part(a) of Big Data Summer 2017

The file is seperated into two parts:
	1) the helper functions
	2) the main driver.

The helper functions are all functions necessary to finish the problem.
The main driver will use the helper functions you finished to report and print
out the results you need for the problem.

First, please COMMENT OUT any steps other than step 0 in main driver before
you finish the corresponding functions for that step. Otherwise, you won't be
able to run the program because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

2. Remember to comment out the TODO comment after you finish each part.
"""

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################
#			 Helper Functions	    	#
#########################################
def get_accuracy(y_pred, y):
	"""	This function takes in two arguments:
			1) y_pred, the predicted label of the data with dimension m x 1
			2) y, the actual label of the data with dimension m x 1

		This function calculates and returns the accuracy of the prediction.

		HINT:
			1) You may use .astype(int) to cast an array into integer type.
	"""
	"*** YOUR CODE HERE ***"
	diff = (y_pred == y).astype(int)
	accu = 1. * diff.sum() / len(y)
	return accu
	"*** END YOUR CODE HERE ***"
	

def get_knn(x, X_train, y_train, k=10):
	"""	This function takes in four arguments:
			1) x, a data vector with dimension of 1 x n
			2) X_train, the data matrix for training data
			3) y_train, the label of the training data
			4) k, the number of the nearest neighbors

		This function finds and returns the index of the k nearest neighbors
		for the given data vector.

		HINT:
			1) You may use sorted() to sort an array by a specified key.

		NOTE: Use l2-norm as the distance metric.
	"""
	"*** YOUR CODE HERE ***"
	m_train = X_train.shape[0]
	dist_list = []
	for index in range(m_train):
		dist_vec = x - X_train[index, :]
		dist = np.linalg.norm(dist_vec)
		dist_list.append((index, dist))
	dist_list = sorted(dist_list, key=lambda x: x[1])[:k] # sort by distance
	return np.array([entry[0] for entry in dist_list]) # returns only the index
	"*** END YOUR CODE HERE ***"
	

def predict_knn(X_test, X_train, y_train, k=10, print_freq=100):
	"""	This function takes in four arguments:
			1) X_test, the data matrix for test data
			2) X_train, the data matrix for training data
			3) y_train, the label of the training data
			4) k, the number of the nearest neighbors
			5) print_freq, the frequency of printing the report

		This function finds the label of k nearest neighbors for the test data.

		HINT:
			1) Use np.bincount() to count the number of appearance for each
				element in the array.
			2) Use .flatten() to flatten an array so that it could fit into np.bincount()
			3) Use np.argmax() to find the index of the maximum element in the
				array.

		NOTE: 
	"""
	m_test = X_test.shape[0]
	m_train = X_train.shape[0]
	y_pred = np.zeros(m_test)
	for index in range(m_test):
		if (index + 1) % print_freq == 0:
			print('Calculating the {}-th data point'.format(index + 1))
		curr = X_test[index, :]
		curr_knn = get_knn(curr, X_train, y_train, k=k)
		y_pred[index] = np.argmax(np.bincount(y_train[curr_knn].flatten()))
	return y_pred.astype(int).reshape(-1, 1)
		


# *****************************************************************
# ====================main driver function: KNN=========================
if __name__ == '__main__':
	# =============STEP 0: LOADING DATA=================
	df_train = data.df_train
	df_test = data.df_test
	'''
		X is a matrix with dimension m x n
		y is a vector with dimension m x 1
		We downsampled the training set to 2500 data points
		and the test set to 500 data points
	'''
	X_train = data.X_train[:2500, :]
	y_train = data.y_train[:2500]
	X_test = data.X_test[:500, :]
	y_test = data.y_test[:500]
	# stacking an array of ones
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_test = np.hstack((np.ones_like(y_test), X_test))
	# =============STEP 1: KNN=================
	# NOTE: Fill in the code in get_knn() and predict_knn() for this step

	# =============STEP 2: Running KNN on Different k values=================
	# NOTE: Fill in the code in get_accuracy() for this step
	k_list = [1, 5, 10]
	for k in k_list:
		# TODO: Find the accuracy for given k values
		"*** YOUR CODE HERE ***"
		y_pred = predict_knn(X_test, X_train, y_train, k=k)
		accu = get_accuracy(y_pred, y_test)
		print('k = {}: accuracy is {: 4.4f}'.format(k, accu))
		"*** END YOUR CODE HERE ***"
