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
variables we providesd, otherwise, the function may not work.

2. Remember to comment out the TODO comment after you finish each part.
"""


#########################################
#			 Helper Functions	    	#
#########################################

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


#########################
#	    Step 1&2		#
#########################

def sigmoid(x):
	'''
		The sigmoid function.
	'''
	return 1. / (1. + np.exp(-x))

def grad_logreg(X, y, W, reg = 0.0):
	'''
		Return the gradient of W for logistic regression.
	'''
	# YOUR CODE BELOW
	return grad

def newton_step(X, y, W, reg = 0.0):
	'''
	Return d, the change of W according to Newton's method.
	'''
	# YOUR CODE BELOW
	return d

def NLL(X, y, W, reg = 0.0):
	'''
		Calculate negative log likelihood.
	'''
	# YOUR CODE GOES BELOW
	return nll

def grad_descent(X, y, reg = 0.0, lr = 1e-4, eps = 1e-6, \
	max_iter = 500, print_freq = 20):
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
	nll_list = []
	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	W_grad = np.ones_like(W)
	print('==> Running gradient descent...')
	iter_num = 0
	t_start = time.time()
	# Running the gradient descent algorithm
	# Update W
	# Calculate learning objectives
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# YOUR CODE GOES BELOW
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - \
				negative log likelihood {: 4.4f}'.format(iter_num + 1, nll))
		iter_num += 1
	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} \
		seconds'.format(t = t_end - t_start))

	return W, nll_list

def newton_method(X, y, reg = 0.0, eps = 1e-6, \
	max_iter = 20, print_freq = 5):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.
		reg is the parameter for regularization.
		eps is the threshold of the norm for the gradients.
		max_iter is the maximum number of iterations.
		print_freq is the frequency of printing the report.

		Return the optimal weight by Netwon's method and the corresponding
		learning objectives.
	'''
	m, n = X.shape
	nll_list = []
	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	step = np.ones_like(W)
	print('==> Running Newton\'s method...')
	iter_num = 0
	t_start = time.time()
	# Running the Newton's method
	# Update W
	# Calculate learning objectives
	while iter_num < max_iter and np.linalg.norm(step) > eps:
		# YOUR CODE GOES BELOW

		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - \
				negative log likelihood {: 4.4f}'.format(iter_num + 1, nll))
		iter_num += 1
	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running Newton\'s method: {t:2.2f} \
		seconds'.format(t = t_end - t_start))

	return W, nll_list




#########################
#		 Step 3			#
#########################

def predict(X, W):
	'''
		Return the predicted labels.
	'''
	mu = sigmoid(X @ W)
	return (mu >= 0.5).astype(int)


def get_description(X, y, W):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.
		W is the weight with dimension (n + 1) x 1.

		Return the accuracy, precision, recall and F-1 score of the prediction.
	'''
	# YOUR CODE GOES BELOW

	return accuracy, precision, recall, f1

def plot_description(X_train, y_train, X_test, y_test):
	'''
		X is matrix with dimension m x (n + 1).
		y is label with dimension m x 1.

		Plot accuracy/precision/recall/F-1 score versus lambda.
		Return the lambda that maximizes accuracy.
	'''
	reg_list = []
	a_list = []
	p_list = []
	r_list = []
	f1_list = []
	# YOUR CODE GOES BELOW

	# Generate plots
	a_vs_lambda_plot, = plt.plot(reg_list, a_list)
	plt.setp(a_vs_lambda_plot, color = 'red')
	p_vs_lambda_plot, = plt.plot(reg_list, p_list)
	plt.setp(p_vs_lambda_plot, color = 'green')
	r_vs_lambda_plot, = plt.plot(reg_list, r_list)
	plt.setp(r_vs_lambda_plot, color = 'blue')
	f1_vs_lambda_plot, = plt.plot(reg_list, f1_list)
	plt.setp(f1_vs_lambda_plot, color = 'yellow')
	plt.legend((a_vs_lambda_plot, p_vs_lambda_plot, r_vs_lambda_plot, \
		f1_vs_lambda_plot), ('accuracy', 'precision', 'recall', 'F-1'),\
		 loc = 'best')
	plt.title('Testing descriptions')
	plt.xlabel('regularization parameter')
	plt.ylabel('Metric')
	plt.savefig('p2a_description.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	# Find the param that maximizes accuracy
	# YOUR CODE GOES BELOW

	return reg_opt




###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet


if __name__ == '__main__':


	# =============STEP 0: LOADING DATA=================
	# NOTE: The data is loaded using the code in p2_data.py. Please make sure
	#		you read the code in that file and understand how it works.

	# data frame
	df_train = data.df_train
	df_test = data.df_test

	# training data
	X_train = data.X_train
	y_train = data.y_train

	# test data
	X_test = data.X_test
	y_test = data.y_test



	# =============STEP 1: Logistic regression=================
	print('==> Step 1: Running logistic regression...')

	# splitting data for logistic regression
	# NOTE: for logistic regression, we only want images with label 0 or 1.
	df_train_logreg = df_train[df_train.label <= 1]
	df_test_logreg = df_test[df_test.label <= 1]

	# training data for logistic regression
	X_train_logreg = np.array(df_train_logreg[:][[col for \
		col in df_train_logreg.columns if col != 'label']]) / 256.
	y_train_logreg = np.array(df_train_logreg[:][['label']])

	# testing data for logistic regression
	X_test_logreg = np.array(df_test_logreg[:][[col for \
		col in df_test_logreg.columns if col != 'label']]) / 256.
	y_test_logreg = np.array(df_test_logreg[:][['label']])

	# stacking a column of 1's to both training and testing data
	X_train_logreg = np.hstack((np.ones_like(y_train_logreg), X_train_logreg))
	X_test_logreg = np.hstack((np.ones_like(y_test_logreg), X_test_logreg))


	# ========STEP 1a: Gradient descent=========
	# NOTE: Fill in the code in NLL and grad_descent for this step

	print('==> Step 1a: Running gradient descent...')
	W_gd, nll_list_gd = grad_descent(X_train_logreg, y_train_logreg, reg = 1e-6)


	# ========STEP 1b: Newton's method==========
	# NOTE: Fill in the code in newton_step and newton_method for this step

	print('==> Step 1b: Running Newton\'s method...')
	W_newton, nll_list_newton = newton_method(X_train_logreg, y_train_logreg, \
		reg = 1e-6)



	# =============STEP 2: Generate convergence plot=================
	# NOTE: You DO NOT need to fill in any additional helper function for this
	# 		step to run. This step uses what you implemented for the previous
	#		two steps to plot.

	print('==> Plotting convergence plot...')

	# set up the style for the plot
	plt.style.use('ggplot')

	# plot gradient descent and newton's method convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	nll_newton_plot, = plt.plot(range(len(nll_list_newton)), nll_list_newton)
	plt.setp(nll_newton_plot, color = 'green')

	# add legend, titles, etc. for the plots
	plt.legend((nll_gd_plot, nll_newton_plot), \
		('Gradient descent', 'Newton\'s method'), loc = 'best')
	plt.title('Convergence Plot on Binary MNIST Classification')
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('p2a_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')



	# =============STEP 3: Generate accuracy/precision plot=================
	# NOTE: Fill in the code in get_description and plot_description for this Step

	print('Step 3: ==> Generating plots for accuracy, precision, recall, and F-1 score...')

	# Plot the graph and obtain the optimal regularization parameter
	reg_opt = plot_description(X_train_logreg, y_train_logreg, \
		X_test_logreg, y_test_logreg)

	print('==> Optimal regularization parameter is {:4.4f}'.format(reg_opt))
