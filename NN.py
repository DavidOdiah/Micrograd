from Engine import Value
import random


class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((w1*x1 for w1, x1 in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def _parameters(self):
        return self.w + [self.b]
    

class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def _parameters(self):
        ret = []
        for neuron in self.neurons:
            ret.extend(neuron._parameters())
        return ret


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _parameters(self):
        ret = []
        for layer in self.layers:
            ret.extend(layer._parameters())
        return ret
    
    def fit(self, xs, ys, epochs=100, learning_rate=0.1):
        # Starting learning rate
        lr = learning_rate

        # Gradient Descent
        epochs = 200
        for i in range(epochs):

            # Forward Parse
            ypred = [self(x) for x in xs]
            loss = sum([(yhat - y)**2 for y, yhat in zip(ys, ypred)], start=Value(0))

            # Backward Parse
            for p in self._parameters():
                p.grad = 0.0
            loss.backward()

            # Update Parameters
            # Making the learning smaller on each iteration of GD
            lr = (learning_rate / 100) * (((epochs - i) / epochs) * 100)
            for p in self._parameters():
                p.data += -lr * p.grad

        print(f"Training Complete - loss: {loss.data}")
    
    def predict(self, x):
        return self(x)

    
# Train with little data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, -0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [-1.0, 1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])
model.fit(xs, ys, epochs=200, learning_rate=0.2)

# Predict with model
x = [1.0, 1.0, -1.0]
print(f"The output of {x} is {model.predict(x)}")