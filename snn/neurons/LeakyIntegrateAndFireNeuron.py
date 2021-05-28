import numpy as np
from snn.neurons.neuron import Neuron

"""
Leaky Integrate and Fire neuron model
"""
class LeakyIntegrateAndFireNeuron(Neuron):
  def __init__(self, weights=0):
    super(self.__class__, self).__init__(weights)
    self.degradation = 0.9
    self.refractoryTime = 0
    self.time = 0
    self.fired_time = 0
    self.time_0 = 0
    self.tau = 10

  def calculate_potential (self, inputs):
    R = 0.9
    I = np.sum(np.multiply(inputs, self.weights))
    if I != 0:
      self.potential = self.potential + I*R
    else:
      self.potential = self.potential + abs(self.potential)*(-0.1)
      if self.potential < self.rest_V:
        self.potential = self.rest_V
    return self.potential

  def solve(self, inputs):
    self.inputs = inputs
    self.time += 1
    if self.inputs.mean() > 0:
      self.time_0 = self.time
    if self.refractoryTime > 0:
      self.refractoryTime = self.refractoryTime - 1
      self.potential = self.rest_V
      self.value = self.rest_V
    else:
      self.value = self.calculate_potential(inputs)
    self.fire()
    if self.fired:
      self.refractoryTime = 2
      self.fired_time = self.time
    return self.fired