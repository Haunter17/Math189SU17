"""
Start file for hw1pr1 of Big Data Summer 2017

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
import time

#########################################
#			 Helper Functions	    	#
#########################################
###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')

	# Read data
	df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', \
		sep=', ', engine='python')
	X = df[[col for col in df.columns if col not in ['url', 'shares', 'cohort']]]
	y = np.log(df.shares).values.reshape(-1,1)
	X = np.hstack((np.ones_like(y), X))
