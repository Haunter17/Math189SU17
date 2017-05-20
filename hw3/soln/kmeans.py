"""
Start file for keans of Big Data Summer 2017

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
import copy
import time
#########################################
#			 Helper Functions	    	#
#########################################

def k_means(X, num_clusters, eps=1e-6, max_iter=1000, print_freq=10):
	"""	This function takes in the following arguments:
			1) X, the data matrix with dimension m x n
			2) num_clusters, the number of clusters
			3) eps, the threshold of the norm of the change in clusters
			4) max_iter, the maximum number of iterations
			5) print_freq, the frequency of printing the report
		
		This function calculates the center of each cluster with the
		k-means algorithm.

		HINT: 

		NOTE: 
	"""
	m, n = X.shape
	"*** YOUR CODE HERE ***"
	clusters = X.std() * np.random.randn(num_clusters, 2) + X.mean()
	label = -1. * np.ones((m, 1))
	iter_num = 0
	while iter_num < max_iter:
		prev_clusters = copy.deepcopy(clusters)
		# find closets center for each data point
		for i in range(m):
			data = X[i, :]
			diff = data - clusters
			label[i] = np.argsort(np.linalg.norm(data - diff, axis=1)).item(0)
		# update centers
		for k in range(num_clusters):
			ind = np.where(label == k)[0]
			clusters[k, :] = X[ind].mean(axis=0)
		if np.linalg.norm(prev_clusters - clusters) <= eps:
			break
		iter_num += 1
	"*** END YOUR CODE HERE ***"

def knn_cost(X, clusters, label):
	"""	This function takes in the following arguments:
			1) X, the data matrix with dimension m x n
			2) clusters, the matrix with dimension k x 1
			3) label, the label of the cluster for each data point with
				dimension m x 1
		
		This function calculates and returns the cost for the given data
		and clusters.

		HINT: 

		NOTE: 
	"""
	m, n = X.shape
	k = clusters.shape[0]
	"*** YOUR CODE HERE ***"
	
	"*** END YOUR CODE HERE ***"


	
###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':
	# =============STEP 0: LOADING DATA=================
	print('==> Step 0: Loading data...')
	# Read data
	path = '../5000_points.csv'
	columns = ['x', 'space', 'y']
	features = ['x', 'y']
	df = pd.read_csv(path, sep='  ', names = columns, engine='python')
	X = np.array(df[:][features]).astype(int)
	# =============STEP 1: Implementing K-MEANS=================
	# NOTE: Fill in the code in k-means()

	# =============STEP 2: FIND OPTIMAL NUMBER OF CLUSTERS=================
	# NOTE: Fill in the code in cost()


