"""
Start file for hw1pr1 of Big Data Summer 2017

Note:

1. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

2. Remember to comment out the TODO comment after you finish each part.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
###########################################
#	    	Main Driver Function       	  #
###########################################
if __name__ == '__main__':

	# =============part c: Plot data and the optimal linear fit=================
	X = np.array([0, 2, 3, 4])
	y = np.array([1, 3, 6, 8])
	plt.style.use('ggplot')
	plt.plot(X, y, 'ro')
	# generate space for optimal linear fit
	m_opt = 62. / 35 # your solution from part a
	b_opt = 18. / 35 # your solution from part a
	X_space = np.linspace(-1, 5, num=100).reshape(-1, 1)
	y_space = np.array([m_opt * x + b_opt for x in X_space]).reshape(-1, 1)
	plt.plot(X_space, y_space)
	plt.savefig('p6c.png', format='png')
	plt.close()

	# =============part d: Optimal linear fit with random data points=================
	# generate random data points
	mu, sigma, sampleSize = 0, 1, 100	
	noice = np.random.normal(mu, sigma, sampleSize)
	y_space_rand = np.zeros(len(X_space))
	for i in range(len(X_space)):
		y_space_rand[i] = m_opt * X_space[i] + b_opt + noice[i]
	# calculate new weights
	X_space_stacked = np.hstack((np.ones_like(y_space), X_space))
	W_opt = np.linalg.solve(X_space_stacked.T @ X_space_stacked, 
		X_space_stacked.T @ y_space_rand)
	b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)
	# calculate predicted values
	y_pred_rand = np.array([m_rand_opt * x + b_rand_opt for x in X_space]).reshape(-1, 1)
	# generate plots with legend
	plt.plot(X, y, 'ro')
	orig_plot, = plt.plot(X_space, y_space, 'r')
	plt.plot(X_space, y_space_rand, 'bo')
	rand_plot, = plt.plot(X_space, y_pred_rand, 'b')
	plt.legend((orig_plot, rand_plot), \
		('original fit', 'fit with noise'), loc = 'best')
	plt.savefig('p6d.png', format='png')
	plt.close()
