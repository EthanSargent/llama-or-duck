import numpy as np
import numpy.linalg as LA
from images import image_manager

DEBUG = True

def sigmoid_unvectorized(x):
	return 1.0/(1.0+np.e**(-x))

sigmoid = np.vectorize(sigmoid_unvectorized)

def cost_grad(nna, images, labels, theta, classes):
	""" returns the cost and gradient associated with a runthrough of the data
		stored in images
	"""
	numHidden = len(nna['layer_sizes']) - 1
	activations = [0] * (numHidden + 2)
	gradStack  = [0] * (numHidden + 1)

	X = images # 784 x 60000
	Y = labels # 60000 x 1
	K = len(classes) # 10
	m = len(X[0]) # 60000

	### forward propagation
	# the activation to the first hidden layer is the unmodified inputs
	activations[0] = X
	a = X
	for l in range(1, numHidden + 2):
		# print(theta)
		W_l = theta[l-1]['W']
		b_l = np.tile(theta[l-1]['b'],(1,m))
		activations[l] = sigmoid(np.dot(W_l, a) + b_l)
		# activations[l] has dimensions layer_l_size x m
		a = activations[l]

	# a Kxm matrix; each column corresponds to all the predictions for a
	# particular example
	pred_prob = a
	# an Kxm array of Kx1 column bit vectors with a 1 indicating the correct class for the ith example
	which_class = np.zeros((K,m))
	# exponentiating but technically the softmax function has not been applied fully, yet...
	softmax_pred_prob = np.exp(pred_prob)
	### phat lewp!! remove if possible
	for i in range(m):
		# making the bit vectors
		which_class[Y[i][0], i] = 1
		# take the L1 norms of the columns
		softmax_pred_prob[:,i] = np.log(softmax_pred_prob[:,i] / LA.norm(softmax_pred_prob[:,i], 1));
	assert which_class[5][0] == 1

	# take the transposes so we can traverse over the rows
	which_class_T = np.transpose(which_class)
	softmax_pred_prob_T = np.transpose(softmax_pred_prob)
	# the total cross entropy cost
	ceCost = - np.sum([np.dot(which_class_T[i], softmax_pred_prob_T[i]) for i in range(m)])
	del which_class_T, softmax_pred_prob_T

	## compute gradients using backpropagation
	# initialize gradStack
	for l in range(0, numHidden + 1):
		# initialize gradients at zero since they're going to be
		# plus-equaled and we want the elementwise operation to do nothing
		gradStack[l] = {'W':0,'b':0}

	# make the Kxm matrix of deltas for each example 
	delta_l_all = - (which_class - pred_prob)
	# calculate gradients for each training example for the weights in each layer
	for i in range(m):
		# Kx1 column vector of deltas for example i
		delta_l = delta_l_all[:,i]
		delta_l.shape = (K,1)
		# starting at the n-1th layer, or last hidden layer, and ending at the last hidden layer
		for l in range(numHidden, 0, -1):
			# storing deltas for layer l + 1
			delta_l1 = delta_l
			delta_l = np.dot(np.transpose(theta[l]['W']), delta_l1)
			# hadamard product of column vectors with f'(z) for the ith example 
			a = activations[l][:,i]; a.shape = (len(a),1)
			f_prime_zl = a - a**2
			delta_l = delta_l * f_prime_zl

			gradStack[l]['W'] += np.dot(delta_l1, np.transpose(a))
			gradStack[l]['b'] += delta_l1

		a = activations[0][:,i]; a.shape = (len(a),1)
		# if DEBUG: print("delta_0: ")
		# print(len(delta_l))
		# print("a_col:")
		# print(len(np.transpose(a)))

		if (i == 0):
			print('delta')
			print(delta_l)
			print('activation')
			print(np.transpose(a))
			print('result')
			result = np.dot(delta_l, np.transpose(a))
			result_sum = np.sum(np.sum(result))
			print(np.dot(delta_l, np.transpose(a)))
			print('result sum')
			print(result_sum)

		# print("shapes")
		# print(delta_l.shape)
		# print(np.transpose(a).shape)
		gradStack[0]['W'] += np.dot(delta_l, np.transpose(a))
		gradStack[0]['b'] += delta_l

	print("weight adjustment for layer 0")
	print(gradStack[0]['W'])
	print("bias adjustment for layer 0")
	print(gradStack[0]['b'])
	return ceCost, gradStack

def grad_descent(theta, gradStack, alpha, m):
	for l in range(len(gradStack)):
		adj_w = - alpha * (1.0/m * gradStack[l]['W'])
		adj_b = - alpha * (1.0/m * gradStack[l]['b'])
		# if DEBUG: print("ADJUSTING WEIGHTS FOR LAYER " + str(l) + " BY: ")
		# if DEBUG: print(adj_w)
		# if DEBUG: print("ADJUSTING BIASES FOR LAYER " + str(l) + " BY: ")
		# if DEBUG: print(adj_b)
		theta[l]['W'] += adj_w # potentially could add weight regularization
		theta[l]['b'] += adj_b
	return theta

def test(nna, images, labels, theta):
	""" tests the NN architecture given by nna and theta on the images and labels
	"""
	m = len(labels)
	guess_vec = guesses(nna, images, labels, theta)
	numCorrect = 0
	for i in range(m):
		if guess_vec[i] == labels[i][0]:
			numCorrect += 1

	print("Percent Correct: %" + str(100.0 * numCorrect / m))

def guesses(nna, images, labels, theta):
	""" computes all the network's digit guesses for a given dataset and stores them
		in a mx1 row vector.
	"""
	numHidden = len(nna['layer_sizes']) - 1
	activations = [0] * (numHidden + 1)
	m = len(labels)
	guess_vec = [0] * m
	a = images
	for l in range(0, numHidden + 1):
		# print(theta)
		W_l = theta[l]['W']
		b_l = np.tile(theta[l]['b'],(1,m))
		activations[l] = sigmoid(np.dot(W_l, a) + b_l)
		# activations[l] has dimensions layer_l_size x m
		a = activations[l]
	
	# get best guess for each image
	for i in range(m):
		best_guess_prob = max(a[:,i])
		guessed_digit = np.where(a[:,i] == best_guess_prob)
		# break ties arbitrarily
		if type(guessed_digit) != int:
			guess_vec[i] = guessed_digit[0][0]
		else:
			guess_vec[i] = guessed_digit

	print("Here are a few guesses: ")
	print(a[:,0])
	print(a[:,1])
	print(a[:,2])

	return guess_vec

	 










