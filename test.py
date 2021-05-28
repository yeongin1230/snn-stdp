from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN
import matplotlib.pyplot as plt

o1 = []
o2 = []
o3 = []
o4 = []
o5 = []
o6 = []
o7 = []
o8 = []

o_spike1 = []
o_spike2 = []
o_spike3 = []
o_spike4 = []
o_spike5 = []
o_spike6 = []
o_spike7 = []
o_spike8 = []


i_spike1 = []
i_spike2 = []
i_spike3 = []
i_spike4 = []
i_spike5 = []
i_spike6 = []
i_spike7 = []
i_spike8 = []

w00 = []
w01 = []
w02 = []
w03 = []
w04 = []
w05 = []
w06 = []
w07 = []

N = range(600)

nn = SNN(3, 8, 8, LeakyIntegrateAndFireNeuron, STDP())

for i in range(3):
    for k in range(200):
        if i == 0:
            final = nn.solve([1, 0, 0])
        elif i == 1:
            final = nn.solve([0, 1, 0])
        elif i == 2:
            final = nn.solve([0, 0, 1])
        i_spike1.append(nn.layers[1][0].fired)
        i_spike2.append(nn.layers[1][1].fired)
        i_spike3.append(nn.layers[1][2].fired)
        i_spike4.append(nn.layers[1][3].fired)
        i_spike5.append(nn.layers[1][4].fired)
        i_spike6.append(nn.layers[1][5].fired)
        i_spike7.append(nn.layers[1][6].fired)
        i_spike8.append(nn.layers[1][7].fired)
        o_spike1.append(nn.layers[2][0].fired)
        o_spike2.append(nn.layers[2][1].fired)
        o_spike3.append(nn.layers[2][2].fired)
        o_spike4.append(nn.layers[2][3].fired)
        o_spike5.append(nn.layers[2][4].fired)
        o_spike6.append(nn.layers[2][5].fired)
        o_spike7.append(nn.layers[2][6].fired)
        o_spike8.append(nn.layers[2][7].fired)
        o1.append(nn.layers[2][0].potential)
        o2.append(nn.layers[2][1].potential)
        o3.append(nn.layers[2][2].potential)
        o4.append(nn.layers[2][3].potential)
        o5.append(nn.layers[2][4].potential)
        o6.append(nn.layers[2][5].potential)
        o7.append(nn.layers[2][6].potential)
        o8.append(nn.layers[2][7].potential)
        w00.append(nn.layers[2][0].weights[0])
        w01.append(nn.layers[2][0].weights[1])
        w02.append(nn.layers[2][0].weights[2])
        w03.append(nn.layers[2][0].weights[3])
        w04.append(nn.layers[2][0].weights[4])
        w05.append(nn.layers[2][0].weights[5])
        w06.append(nn.layers[2][0].weights[6])
        w07.append(nn.layers[2][0].weights[7])


# plt.figure(figsize=(10,5))
# plt.subplot(6,2,1)
# plt.bar(N, i_spike1)
# plt.subplot(6,2,3)
# plt.bar(N, i_spike2)
# plt.subplot(6,2,5)
# plt.bar(N, i_spike3)
# plt.subplot(6,2,7)
# plt.bar(N, i_spike4)
# plt.subplot(6,2,9)
# plt.bar(N, i_spike5)
# plt.subplot(6,2,11)
# plt.bar(N, i_spike6)
plt.figure(figsize=(20,8))
plt.subplot(8,4,1)
plt.bar(N, i_spike1)
plt.subplot(8,4,5)
plt.bar(N, i_spike2)
plt.subplot(8,4,9)
plt.bar(N, i_spike3)
plt.subplot(8,4,13)
plt.bar(N, i_spike4)
plt.subplot(8,4,17)
plt.bar(N, i_spike5)
plt.subplot(8,4,21)
plt.bar(N, i_spike6)
plt.subplot(8,4,25)
plt.bar(N, i_spike7)
plt.subplot(8,4,29)
plt.bar(N, i_spike8)

plt.subplot(8,4,2)
plt.plot(N,w00,'.c')
plt.subplot(8,4,6)
plt.plot(N,w01,'.c')
plt.subplot(8,4,10)
plt.plot(N,w02,'.c')
plt.subplot(8,4,14)
plt.plot(N,w03,'.c')
plt.subplot(8,4,18)
plt.plot(N,w04,'.c')
plt.subplot(8,4,22)
plt.plot(N,w05,'.c')
plt.subplot(8,4,26)
plt.plot(N,w06,'.c')
plt.subplot(8,4,30)
plt.plot(N,w07,'.c')

plt.subplot(8,4,3)
plt.plot(N,o1,'.c')
plt.subplot(8,4,7)
plt.plot(N,o2,'.c')
plt.subplot(8,4,11)
plt.plot(N,o3,'.c')
plt.subplot(8,4,15)
plt.plot(N,o4,'.c')
plt.subplot(8,4,19)
plt.plot(N,o5,'.c')
plt.subplot(8,4,23)
plt.plot(N,o6,'.c')
plt.subplot(8,4,27)
plt.plot(N,o7,'.c')
plt.subplot(8,4,31)
plt.plot(N,o8,'.c')

plt.subplot(8,4,4)
plt.bar(N, o_spike1)
plt.subplot(8,4,8)
plt.bar(N, o_spike2)
plt.subplot(8,4,12)
plt.bar(N, o_spike3)
plt.subplot(8,4,16)
plt.bar(N, o_spike4)
plt.subplot(8,4,20)
plt.bar(N, o_spike5)
plt.subplot(8,4,24)
plt.bar(N, o_spike6)
plt.subplot(8,4,28)
plt.bar(N, o_spike7)
plt.subplot(8,4,32)
plt.bar(N, o_spike8)
plt.show()
# plt.subplot(6,2,2)
# plt.plot(N,o1,'.c')
# plt.subplot(6,2,4)
# plt.plot(N,o2,'.c')
# plt.subplot(6,2,6)
# plt.plot(N,o3,'.c')
# plt.subplot(6,2,8)
# plt.bar(N, o_spike1)
# plt.subplot(6,2,10)
# plt.bar(N, o_spike2)
# plt.subplot(6,2,12)
# plt.bar(N, o_spike3)
# plt.show()

plt.plot(nn.learning.delta_t,nn.learning.delta_w,'r.')
plt.show()

