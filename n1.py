from math import tanh
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = []
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return tanh(activation)  
	'''if(activation > 0):
		return activation
	else:
		return 0.0001*activation
	''' 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return (1.0 - output*output)
	'''if(output > 0):
		return 1.0
	else:
		return 0.0001'''
	

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = []
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs,output_labels,matrix):
	for epoch in range(n_epoch):
		sum_error = 0
		index = 0;
		for row in train:
			print('im training\n')
			outputs = forward_propagate(network, row)
			#expected = [0 for i in range(n_outputs)]
			#expected[row[-1]] = 1
			print('output of neural network is ',end = " " )
			print(outputs)

			expected = matrix[output_labels[index]]

			print(expected)
			index += 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))




# Test training backprop algorithm
seed(1)

import numpy as np;
import gzip

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
            dataset = np.frombuffer(f.read(), np.uint8, offset=16)

dataset = dataset.reshape(60000,28*28)

dataset = dataset/np.float32(256)

data = []
for i in range(50):
	data.append(dataset[i])


n_inputs = 28*28
n_outputs = 10


with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)



matrix = []
for i in range(10):
	list = [0 for j in range(10)]
	list[i] = 1
	matrix.append(list)

network = initialize_network(n_inputs, 15, n_outputs)

#print(network)

train_network(network, data, 0.5, 500, n_outputs,labels,matrix)

#for layer in network:
#	print(layer)