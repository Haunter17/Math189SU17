"""
Starter file for nmf of Big Data Summer 2017

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
	3) Let k be the number of topics

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from sklearn.feature_extraction import text

#########################################
#			 Helper Functions	    	#
#########################################
def nmf_cost(X, W, H):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x n
			2) W, a matrix with dimension m x k
			3) H, a matrix with dimension k x n

		This function calculates and returns the cost defined by
		|X - WH|^2.

		HINT: 
	"""
	return np.linalg.norm(X - W @ H)
	# cost = 0.
	# sparse_X = X.tocoo()
	# for i, j, x in zip(sparse_X.row, sparse_X.col, sparse_X.data):
	# 	cost += (x - np.inner(W[i], H[:, j])) ** 2
	# return cost

def nmf(X, k=20, max_iter=100, print_freq=5):
	m, n = X.shape
	W = np.abs(np.random.randn(m, k) * 1e-3)
	H = np.abs(np.random.randn(k, n) * 1e-3)

	cost_list = [nmf_cost(X, W, H)]
	for iter_num in range(max_iter):
		H = H * (W.T @ X) / ((W.T @ W) @ H)
		W = W * (X @ H.T) / (W @ (H @ H.T))
		cost = nmf_cost(X, W, H)
		cost_list.append(cost)
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - cost: {:.4E}'.format(iter_num + 1, \
				cost))

	return W, H.T, cost_list

###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')
	# NOTE: Run nltk.download() in your python shell to download
	# the reuters dataset under Corpora tab
	X = np.array([' '.join(list(reuters.words(file_id))).lower() \
		for file_id in reuters.fileids()])
	tfidf = text.TfidfVectorizer()
	X = tfidf.fit_transform(X)
	# =============STEP 1: RUNNING NMF=================
	# NOTE: Fill in code in nmf_cost(), nmf() for this step
	print('==> Running nmf algorithm on the dataset...')
	W, H, cost_list = nmf(X, print_freq=1)
