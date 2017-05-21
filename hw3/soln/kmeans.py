"""
Starter file for k-means of Big Data Summer 2017

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

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

After finishing the helper functions for each step, you can uncomment
the code in main driver to check the result.

Note:
1. When filling out the functions below, remember to
	1) Let m be the number of samples
	2) Let n be the number of features
	3) Let k be the number of clusters

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

def k_means(X, k, eps=1e-6, max_iter=1000, print_freq=10):
	"""	This function takes in the following arguments:
			1) X, the data matrix with dimension m x n
			2) k, the number of clusters
			3) eps, the threshold of the norm of the change in clusters
			4) max_iter, the maximum number of iterations
			5) print_freq, the frequency of printing the report
		
		This function returns the following:
			1) clusters, a list of clusters with dimension k x 1
			2) label, the label of cluster for each data with dimension m x 1
			3) cost_list, a list of costs at each iteration

		HINT: 

		NOTE: 
	"""
	m, n = X.shape
	cost_list = []
	t_start = time.time()
	"*** YOUR CODE HERE ***"
	mu_x, std_x = X[:, 0].mean(), X[:, 0].std()
	mu_y, std_y = X[:, 1].mean(), X[:, 1].std()
	clusters_x = np.random.normal(mu_x, std_x, size=(k, 1))
	clusters_y = np.random.normal(mu_y, std_y, size=(k, 1))
	clusters = np.hstack((clusters_x, clusters_y))
	label = np.zeros((m, 1)).astype(int)
	iter_num = 0
	while iter_num < max_iter:
		prev_clusters = copy.deepcopy(clusters)
		# find closets center for each data point
		for i in range(m):
			data = X[i, :]
			diff = data - clusters
			curr_label = np.argsort(np.linalg.norm(diff, axis=1)).item(0)
			label[i] = curr_label
		# update centers
		for cluster_num in range(k):
			ind = np.where(label == cluster_num)[0]
			if len(ind) > 0:
				clusters[cluster_num, :] = X[ind].mean(axis=0)
		# calculate costs
		cost = k_means_cost(X, clusters, label)
		cost_list.append(cost)
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - cost {:4.4E}'.format(iter_num + 1, cost))
		if np.linalg.norm(prev_clusters - clusters) <= eps:
			print('-- Algorithm converges at iteration {} \
				with cost {:4.4E}'.format(iter_num + 1, cost))
			break
		iter_num += 1
	"*** END YOUR CODE HERE ***"
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} \
		seconds'.format(t=t_end - t_start))
	return clusters, label, cost_list

def k_means_cost(X, clusters, label):
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
	X_cluster = clusters[label.flatten()]
	cost = (np.linalg.norm(X - X_cluster, axis=1) ** 2).sum()
	"*** END YOUR CODE HERE ***"
	return cost

	
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
	# TODO: Fill in the code in k_means() and k_means_cost()
	# NOTE: You may test your implementations by running k_means(X, k)
	# 		for any reasonable value of k

	# =============STEP 2: FIND OPTIMAL NUMBER OF CLUSTERS=================
	# TODO: calculate the cost for k between 1 and 20 and find the k with
	# 		optimal cost
	cost_k_list = []
	for k in range(1, 21):
		"*** YOUR CODE HERE ***"
		clusters, label, cost_list = k_means(X, k)
		cost = cost_list[-1]
		cost_k_list.append(cost)
		"*** END YOUR CODE HERE ***"
		print('-- Number of clusters: {} - cost: {:.4E}'.format(k, cost))
	opt_k = np.argmin(cost_k_list) + 1
	print('-- Optimal number of clusters is {}'.format(opt_k))
	# TODO: Generate plot of cost vs k
	"*** YOUR CODE HERE ***"
	cost_vs_k_plot, = plt.plot(range(1, 21), cost_k_list, 'g^')
	opt_cost_plot, = plt.plot(opt_k, min(cost_k_list), 'rD')
	"*** END YOUR CODE HERE ***"
	plt.title('Cost vs Number of Clusters')
	plt.savefig('kmeans_1.png', format='png')
	plt.close()
	# =============STEP 3: VISUALIZATION=================
	# TODO: Generate visualization on running k-means on the optimal k value
	# NOTE: Be sure to mark the cluster centers from the data point
	"*** YOUR CODE HERE ***"
	clusters, label, cost_list = k_means(X, opt_k)
	X_cluster = clusters[label.flatten()]
	data_plot, = plt.plot(X[:, 0], X[:, 1], 'bo')
	center_plot, = plt.plot(X_cluster[:, 0], X_cluster[:, 1], 'rD')
	"*** END YOUR CODE HERE ***"
	# set up legend and save the plot to the current folder
	plt.legend((data_plot, center_plot), \
		('data', 'clusters'), loc = 'best')
	plt.title('Visualization with {} clusters'.format(opt_k))
	plt.savefig('kemans_2.png', format='png')
	plt.close()
