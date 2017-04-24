import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# *****************************************************************
# ====================helper functions=========================
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
	return X.T @ (sigmoid(X @ W) - y) + reg * W

def newton_step(X, y, W, reg = 0.0):
	''' 
	Return the change of W according to Newton's method.
	'''
	# YOUR CODE BELOW
	mu = sigmoid(X @ W)
	g = grad_logreg(X, y, W, reg = reg)
	diag = np.diag(np.squeeze(np.asarray(np.multiply(mu, 1. - mu))))
	H = X.T @ diag @ X + reg * np.eye(X.shape[1])
	d = np.linalg.solve(H, g)
	return d

def NLL(X, y, W, reg = 0.0):
	'''
		Calculate negative log likelihood.
	'''
	# YOUR CODE GOES BELOW
	mu = sigmoid(X @ W)
	temp = np.multiply(y, np.log(mu)) + np.multiply((1. - y), np.log(1. - mu))
	nll = sum(temp) - reg / 2 * np.linalg.norm(W) ** 2
	return nll.item(0)

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
	# YOUR CODE GOES BELOW
	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# calculate NLL
		nll = NLL(X, y, W, reg = reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)
		# calculate gradients and update W
		W_grad = grad_logreg(X, y, W, reg = reg)
		W -= lr * W_grad
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
	# YOUR CODE GOES BELOW
	while iter_num < max_iter and np.linalg.norm(step) > eps:
		# calculate NLL
		nll = NLL(X, y, W, reg = reg)
		if np.isnan(nll):
			break
		nll_list.append(nll)
		# calculate gradients and update W
		step = newton_step(X, y, W, reg = reg)
		W -= step
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - \
				negative log likelihood {: 4.4f}'.format(iter_num + 1, nll))
		iter_num += 1
	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running Newton\'s method: {t:2.2f} \
		seconds'.format(t = t_end - t_start))

	return W, nll_list

# *****************************************************************
# ====================main driver function=========================
if __name__ == '__main__':
	# =============STEP 0: LOADING DATA=================
	print('==>Loading data...')
	name_list = ['pix_{}'.format(i + 1) for i in range(784)]
	name_list = ['label'] + name_list
	df_train = pd.read_csv('http://pjreddie.com/media/files/mnist_train.csv', \
		sep=',', engine='python', names = name_list)
	df_test = pd.read_csv('http://pjreddie.com/media/files/mnist_test.csv', \
		sep=',', engine='python', names = name_list)
	print('==>Data loaded succesfully.')
	X_train = np.array(df_train[:][[col for col in df_train.columns \
	if col != 'label']]) / 256.
	y_train = np.array(df_train[:][['label']])
	X_test = np.array(df_test[:][[col for col in df_test.columns \
	if col != 'label']]) / 256.
	y_test = np.array(df_test[:][['label']])

	# =============STEP 1: Logistic regression=================
	print('==> Step 1: Running logistic regression...')
	# splitting data for logistic regression
	df_train_logreg = df_train[df_train.label <= 1]
	X_train_logreg = np.array(df_train_logreg[:][[col for \
		col in df_train_logreg.columns if col != 'label']]) / 256.
	y_train_logreg = np.array(df_train_logreg[:][['label']])
	df_test_logreg = df_test[df_test.label <= 1]
	X_test_logreg = np.array(df_test_logreg[:][[col for \
		col in df_test_logreg.columns if col != 'label']]) / 256.
	y_test_logreg = np.array(df_test_logreg[:][['label']])
	# stacking a column of 1's
	X_train_logreg = np.hstack((np.ones_like(y_train_logreg), X_train_logreg))
	X_test_logreg = np.hstack((np.ones_like(y_test_logreg), X_test_logreg))
	# =============STEP 1a: Gradient descent=================
	print('==> Step 1a: Running gradient descent...')
	# Fill in the code in NLL and grad_descent
	W_gd, nll_list_gd = grad_descent(X_train_logreg, y_train_logreg, reg = 1e-6)
	# =============STEP 1b: Newton's method=================
	# Fill in the code in newton_step and newton_method
	print('==> Step 1b: Running Newton\'s method...')
	W_newton, nll_list_newton = newton_method(X_train_logreg, y_train_logreg, \
		reg = 1e-6)
	# Generate the convergence plot
	print('==> Printing convergence plot...')
	plt.style.use('ggplot')
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')
	nll_newton_plot, = plt.plot(range(len(nll_list_newton)), nll_list_newton)
	plt.setp(nll_newton_plot, color = 'green')
	plt.legend((nll_gd_plot, nll_newton_plot), \
		('Gradient descent', 'Newton\'s method'), loc = 'best')
	plt.title('Convergence Plot on Binary MNIST Classification')
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('p2a_convergence.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')
