import numpy as np
from images import image_manager
import multilayer_neural_network as mlnn

def init_rand(rows, cols, epsilon = 1e-4):
	""" initializes a numpy array of #rows = rows, #cols = cols, where each
		entry is a normally distributed value with mean 0 and SD epsilon.
	"""
	out = np.empty([rows, cols])

	for row in range(rows):
		for col in range(cols):
			out[row][col] = np.random.normal(0,epsilon)

	return out

### most of the boilerplate code for training, initializing a vanilla
### feedforward neural network. adapted from UFLDL Matlab code.

# classes denotes the types of digits we are interested in identifying.
# we want all 10
classes = list(range(10))
images, labels = image_manager.load_mnist("training", classes, "images/")

# learning rate
alpha = .1

# neural network architecture
nna = {}
nna['input_dim'] = 784
nna['output_dim'] = 10
# one hidden layer with 256 neurons
nna['layer_sizes'] = [256, nna['output_dim']]

# for future support of various activations functions
nna['activations_fun'] = 'logistic'

# weights and biases. will be reshaped into a stack structure
theta = [0] * len(nna['layer_sizes'])

theta[0] = {'W':init_rand(nna['layer_sizes'][0], nna['input_dim']), 'b':init_rand(nna['layer_sizes'][0], 1)}
			
for i in range(1, len(nna['layer_sizes'])):
	# initialize the weights and biases from layer i to layer i + 1
	# 0th layer is the inputs
	theta[i] = {'W':init_rand(nna['layer_sizes'][i], nna['layer_sizes'][i-1]), 'b':init_rand(nna['layer_sizes'][i], 1)}

print(len(theta))

epochs = 3
for epoch in range(epochs):
	print("THE EPOCH IS: " + str(epoch))
	f,g = mlnn.cost_grad(nna, images, labels, theta, classes)
	theta = mlnn.grad_descent(theta, g, alpha, len(images[0]))

images, labels = image_manager.load_mnist("testing", classes, "images/")

mlnn.test(nna, images, labels, theta)





