"""
Start file for knn of Big Data Summer 2017

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
	# TODO: Find the accuracy of a prediction
	"*** YOUR CODE HERE ***"


	"*** END YOUR CODE HERE ***"
	return accu
	

def get_knn(x, X_train, y_train, k=10):
	"""	This function takes in four arguments:
			1) x, a data vector with dimension of 1 x n
			2) X_train, the data matrix for training data
			3) y_train, the label of the training data
			4) k, the number of the nearest neighbors

		This function finds and returns the index of the k nearest neighbors
		for the given data vector.

		HINT:
			1) Use np.argsort() to find the indices of a sorted array

		NOTE: Use l2-norm as the distance metric.
	"""
	# TODO: Find the indicies of k nearest neighbors of a data point x
	"*** YOUR CODE HERE ***"


	"*** END YOUR CODE HERE ***"
	return ind[:k]
	

def predict_knn(X_test, X_train, y_train, k=10, print_freq=100):
	"""	This function takes in four arguments:
			1) X_test, the data matrix for test data
			2) X_train, the data matrix for training data
			3) y_train, the label of the training data
			4) k, the number of the nearest neighbors
			5) print_freq, the frequency of printing the report

		This function finds the label of k nearest neighbors for the test data.

		HINT:
			1) Call get_knn() on each data point.
			2) Use np.bincount() to count the number of appearance for each
				element in the array.
			3) Use .flatten() to flatten an array so that it could fit into np.bincount()
			4) Use np.argmax() to find the index of the maximum element in the
				array.
	"""
	m_test = X_test.shape[0]
	m_train = X_train.shape[0]
	y_pred = np.zeros(m_test)

	# TODO: Find the k nearest neighbors for each data point
	for index in range(m_test):
		if (index + 1) % print_freq == 0:
			print('Calculating the {}-th data point'.format(index + 1))

		"*** YOUR CODE HERE ***"
		

		"*** END YOUR CODE HERE ***"

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

	# =============STEP 1: KNN=================
	# NOTE: Fill in the code in get_knn() and predict_knn() for this step

	# =============STEP 2: Running KNN on Different k values=================
	# NOTE: Fill in the code in get_accuracy() for this step
	k_list = [1, 5, 10]
	for k in k_list:
		# TODO: Find the accuracy for given k values
		"*** YOUR CODE HERE ***"


		"*** END YOUR CODE HERE ***"
		print('k = {}: accuracy is {: 4.4f}'.format(k, accu))
