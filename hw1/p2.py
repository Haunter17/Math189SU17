import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
	if col != 'label']])
	y_train = np.array(df_train[:][['label']])
	X_test = np.array(df_test[:][[col for col in df_test.columns \
	if col != 'label']])
	y_test = np.array(df_test[:][['label']])

	# =============STEP 1: Logistic regression=================
	# splitting data for logistic regression
	df_train_logreg = df_train[df_train.label <= 1]
	X_train_logreg = np.array(df_train_logreg[:][[col for \
		col in df_train_logreg.columns if col != 'label']])
	y_train_logreg = np.array(df_train_logreg[:][['label']])
	df_test_logreg = df_test[df_test.label <= 1]
	X_test_logreg = np.array(df_test_logreg[:][[col for \
		col in df_test_logreg.columns if col != 'label']])
	y_test_logreg = np.array(df_test_logreg[:][['label']])
	
