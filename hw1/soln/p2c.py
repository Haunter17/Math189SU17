import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# *****************************************************************
# ====================helper functions=========================
def get_knn(x, X_train, y_train, k=10):
	"""	This function takes in four arguments:
			1) x, a data vector with dimension of 1 x n
			2) X_train, the data matrix for training data
			3) y_train, the label of the training data
			4) k, the number of the nearest neighbors

		This function finds the k nearest neighbors for the given data vector.

		HINT:
			1) 

		NOTE: 
	"""
	m_train = X_train.shape[0]
	dist_list = []
	for index in range(m_train):
		dist_vec = x - X_train[index, :]
		dist = np.linalg.norm(dist_vec)
		dist_list.append((index, dist))
	dist_list = sorted(dist_list, key=lambda x: x[1])[:k]
	return np.array([entry[0] for entry in dist_list])

def predict_knn(X_test, X_train, y_train, k=10, print_freq=1000):
	"""	This function takes in four arguments:
			1) X_test, the data matrix for test data
			2) X_train, the data matrix for training data
			3) y_train, the label of the training data
			4) k, the number of the nearest neighbors
			5) print_freq, the frequency of printing the report

		This function finds the k nearest neighbors for the test data.

		HINT:
			1) 

		NOTE: 
	"""
	m_test = X_test.shape[0]
	m_train = X_train.shape[0]
	y_pred = np.zeros(m_test)
	for index in range(m_test):
		curr = X_test[index, :]
		
		


# *****************************************************************
# ====================main driver function: KNN=========================
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
