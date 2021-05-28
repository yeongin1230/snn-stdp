import numpy as np
from snn.learning.method import LearningMethod

"""
Basic STDP Learning Method
"""
class STDP(LearningMethod):
  def __init__(self):
    self.adjustment = 0.1
    #self.time = 0
    self.is_setup = False
    self.tau = 10
    self.delta_t = []
    self.delta_w = []

  def update_weights(self, neuron, adjustment):
    adjustments = np.multiply(neuron.inputs, adjustment)
    signed_adjustments = np.multiply(adjustments, np.sign(neuron.weights))
    adjusted = np.add(neuron.weights, signed_adjustments)
    return adjusted

  def update_trace(self, neuron):
    # trace = []
    # for (i, x) in enumerate(neuron.inputs):
    #   trace = np.append(trace,0 if x == 0 else neuron.trace[int(i)] + 1)
    trace = [ 0 if x == 0 else neuron.trace[int(i)] + 1 for (i, x) in enumerate(1 - neuron.inputs)]
    return trace

  def setup(self, neuron):
    neuron.trace = np.zeros(len(neuron.inputs))

  def update_(self, layers):
    for i in range(1,len(layers)-1):
      for j in range(len(layers[i])):
        for k in range(len(layers[i - 1])):
          delta_t = layers[i-1][k].fired_time - layers[i][j].fired_time
          if (delta_t > 0):
            delta_w = 0.01*np.exp(-delta_t/self.tau)
          elif (delta_t < 0):
            delta_w = -0.01*np.exp(delta_t/self.tau)
          else:
            delta_w = 0
          layers[i][j].weights[k] += delta_w
          self.delta_t.append(delta_t)
          self.delta_w.append(delta_w)

  def update(self, layers):
    for i in range(1,len(layers)-1):
      for j in range(len(layers[i])):
        if layers[i][j].fired:
          for k in range(len(layers[i - 1])):
            delta_t = layers[i - 1][k].fired_time - layers[i][j].fired_time
            if (delta_t > 0):
              delta_w = -0.015*np.exp(-delta_t/self.tau)
            elif (delta_t <= 0):
              delta_w = 0.015*np.exp(delta_t/self.tau)
            else:
              delta_w = 0
            layers[i][j].weights[k] += delta_w
            self.delta_t.append(delta_t)
            self.delta_w.append(delta_w)

          # for k in range(len(layers[i + 1])):
          #   delta_t = layers[i][j].fired_time - layers[i+1][k].fired_time
          #   if (delta_t > 0):
          #     delta_w = -0.015*np.exp(-delta_t/self.tau)
          #   elif (delta_t < 0):
          #     delta_w = 0.01*np.exp(delta_t/self.tau)
          #   else:
          #     delta_w = 0
          #   layers[i+1][k].weights[j] += delta_w
          #   self.delta_t.append(delta_t)
          #   self.delta_w.append(delta_w)

    for j in range(len(layers[-1])):
      if layers[-1][j].fired:
        for k in range(len(layers[-2])):
          delta_t = layers[-2][k].fired_time - layers[-1][j].fired_time
          if (delta_t > 0):
            delta_w = -0.015 * np.exp(-delta_t / self.tau)
          elif (delta_t <= 0):
            delta_w = 0.015 * np.exp(delta_t / self.tau)
          else:
            delta_w = 0
          layers[-1][j].weights[k] += delta_w
          self.delta_t.append(delta_t)
          self.delta_w.append(delta_w)

          # 1
          # delta_t = layers[i-1][k].fired_time - layers[i][j].fired_time
          # if (delta_t > 0):
          #   delta_w = 0.01*np.exp(-delta_t/self.tau)
          # elif (delta_t < 0):
          #   delta_w = -0.01*np.exp(delta_t/self.tau)
          # else:
          #   delta_w = 0

          #2
          # if layers[i][j].fired_time == layers[i][j].time:
          #   delta_t = layers[i][j].fired_time - layers[i-1][k].fired_time
          #   if (delta_t > 0) and (delta_t < layers[i][j].time):
          #     delta_w = 0.01*np.exp(-delta_t/self.tau)
          #   elif (delta_t < 0) or (delta_t == layers[i][j].time):
          #     delta_w = -0.01*np.exp(delta_t/self.tau)
          #   else:
          #     delta_w = 0
            # delta_w = np.where(delta_t>=0,0.01*np.exp(-delta_t/self.tau),-0.015*np.exp(delta_t/self.tau))
    #3
    # for layer in layers:
    #   for neuron in layer:
    #     if (not self.is_setup):
    #       self.setup(neuron)
    #     correlated_adjustment = self.adjustment if neuron.refractoryTime == 0 else -self.adjustment
    #     neuron.weights = self.update_weights(neuron, correlated_adjustment)
    #     neuron.trace = self.update_trace(neuron)
    # if (not self.is_setup): self.is_setup = True
    #self.time = self.time + 1

