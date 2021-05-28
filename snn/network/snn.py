import numpy as np

"""
Spiking Neural Network model
"""
class SNN:
  def __init__(self, num_input, hidden_layers, num_output, neuron_class, learning_method):
    self.layers = []
    self.neuronClass = neuron_class
    self.learning = learning_method
    self.setup(num_input, hidden_layers, num_output)

  def setup(self, num_input, hidden_layers, num_output):
    # self.setup_layer(num_input)
    self.setup_input_layer(num_input)
    self.setup_hidden(hidden_layers)
    self.setup_layer(num_output)

  def setup_input_layer(self, num_neurons):
    layer_neurons = np.array([])
    for x in range(num_neurons):
      input_weights = num_neurons
      layer_neurons = np.append(layer_neurons,self.neuronClass(input_weights))
    self.layers.append(layer_neurons)
    # input neuron weights initialize -> ex)1st neuron->[1,0,0,...]
    for x in range(num_neurons):
      for y in range(num_neurons):
        if x == y:
          self.layers[0][x].weights[y] = 1
        else:
          self.layers[0][x].weights[y] = 0


  def setup_layer(self, num_neurons):
    layer_neurons = np.array([])
    for x in range(num_neurons):
      # input_weights = len(self.layers[-1]) if len(self.layers) > 0 else num_neurons
      input_weights = len(self.layers[-1])
      layer_neurons = np.append(layer_neurons, self.neuronClass(input_weights))
    self.layers.append(layer_neurons)

  def setup_hidden(self, hidden_layers):
    if type(hidden_layers) is int:
      self.setup_layer(hidden_layers)
    else:
      for layer in hidden_layers:
        self.setup_layer(layer)

  def adjust_weights(self):
    self.learning.update(self.layers)

  def solve(self, input):
    previous_layer = np.array(input)
    for (i, layer) in enumerate(self.layers):
      new_previous_layer = np.array([])
      for neuron in layer:
        new_previous_layer = np.append(new_previous_layer, neuron.solve(previous_layer))
      previous_layer = new_previous_layer
    self.adjust_weights()
    self.inhibition()
    return previous_layer

  def inhibition(self):
    for i in range(len(self.layers)):
      firelist = []
      for j in range(len(self.layers[i])):
        if self.layers[i][j].fired:
          firelist.append(1)
        else:
          firelist.append(0)
      if 1 in firelist:
        max_V = 0
        max_index = 0
        for j in range(len(self.layers[i])):
          if self.layers[i][j].potential > max_V:
            max_V = self.layers[i][j].potential
            max_index = j
        for j in range(len(self.layers[i])):
          if j != max_index:
            self.layers[i][j].potential = self.layers[i][j].inhi_V
            self.layers[i][j].value = self.layers[i][j].inhi_V

  def inhibition_(self):
    fired = [0]*len(self.layers)
    for (i, layer) in enumerate(self.layers):
      for neuron in layer:
        if neuron.fired:
          fired[i] = 1
    for (i, fire) in enumerate(fired):
      if fire == 1:
        for neuron in self.layers[i]:
          neuron.refractoryTime = 2

