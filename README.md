USED LIBRARY:
- maths
- random


CREATING MICROGRAD:
Create our Engine:
	Value class.
		- Houses our data for each value
		- Should be able to perform the operations needed for our value object to be used in a Neural Network.
		- Should have references to the value objects that gave result to itself if it is a result of an operation between one or two value objects.
		- Should be represented.
		- Should be able to performback propagation.
     
Create our Neural Network:
	Neuron class:
		- Houses our weight and Bias Values in the neuron
		- Should pass our input through our Neuron and the output through our activation function when called on an input data.

	Layer Class:
		- Houses our list of Neurons on that layer.
		- Should pass our input data through every neuron in the layer when called, including our activation function.

	Multi-Layer Perceptron (MLP) Class:
		- Houses all the layers in our Neural Network.
		- Should pass our input data through the Neural Network when called.