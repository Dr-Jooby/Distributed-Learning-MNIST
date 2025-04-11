import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct

from Network import Network
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from ActivationFunction import tanh, tanh_prime
from LossFunction import mse, mse_prime, accuracy
from LoadData import load_mnist_images, load_mnist_labels, load_data_IID, load_data_non_IID
from Agent import Agent
from NetworkTopology import Topology


x_train, x_test, y_train, y_train_new, y_test,x = load_data_IID()


batch_size_accuracy = [[],[],[],[],[]]
control_batch_size_accuracy = []

#for batch_list in range(400):
#print('Currently in Batch Number %d' % (batch_list))

agent_control = Agent()
agent = [[],[],[],[],[]]
agent_loss = [[],[],[],[],[]]
control_train = []
control_test = []
a_control = []

training_accuracy = []

testing_accuracy = []
control_testing_accuracy = []


Learning_Rate = 0.01

batches = 400 * 10 + 1
control_batch_size = 14 * 5#len(y_train)/batches

batch_size = list()

for i in range(len(y_train_new)):
    batch_size.append(14)#len(y_train_new[i])/batches)


for i in range(len(agent)):
    training_accuracy.append([])
    testing_accuracy.append([])

for i in range(5):
    agent[i] = Agent()
for i in range(5):
    agent[i].predict(x_test, y_test)
    agent_loss[i].append(agent[i].net.test_acc)
    training_accuracy[i].append(agent[i].net.test_acc)
    testing_accuracy[i].append(agent[i].net.test_acc)


agent_control.predict(x_test,y_test)
a_control.append(agent_control.net.test_acc)

for i in range(batches):
    Selection = np.random.randint(0,800)
    # start = int(i*control_batch_size)
    # end = int((i+1)*control_batch_size-1)
    start = int(Selection*control_batch_size)
    end = int((Selection+1)*control_batch_size-1)
    agent_control.fit_train(x_train[start:end], y_train[start:end], epochs=1, learning_rate=Learning_Rate)
    control_train.append(agent_control.net.train_acc)

    if i%40 == 0:
            agent_control.predict(x_test, y_test)
            control_testing_accuracy.append(agent_control.net.test_acc)

a_control.append(agent_control.net.train_acc)
agent_control.predict(x_test,y_test)
control_batch_size_accuracy.append(agent_control.net.test_acc)


for j in range(batches):
    Topology(agent, "Ring")
    for i in range(5):
        # start = int(j*batch_size[i])
        # end = int((j+1)*batch_size[i]-1)
        Selection = np.random.randint(0,800)
        start = int(Selection*batch_size[i])
        end = int((Selection+1)*batch_size[i]-1)

        agent[i].fit_train(x[i][start:end], y_train_new[i][start:end], epochs=1, learning_rate=Learning_Rate)
        training_accuracy[i].append(agent[i].net.train_acc)
    
        if j%40 == 0:
            agent[i].predict(x_test, y_test)
            testing_accuracy[i].append(agent[i].net.test_acc)
            if i == 2:
                print('Accuracy = %f || Currently in Training Round %d / 4000' % (testing_accuracy[i][-1], j))


for i in range(5):
    agent_loss[i].append(agent[i].net.train_acc)
    agent[i].predict(x_test, y_test)
    batch_size_accuracy[i].append(agent[i].net.test_acc)


# plt.figure(0)
# plt.plot(np.arange(1,len(a_control)+1,1), a_control, label = "Centralized")

# plt.figure(1)
# plt.plot(np.arange(1,len(control_train)+1,1), control_train, label = "Centralized")

plt.figure(2)
plt.plot(np.arange(1,40*len(control_testing_accuracy)+1,40), control_testing_accuracy, label = "Centralised")

for i in range(5):  
    # plt.figure(0)
    # plt.plot(np.arange(1,len(agent_loss[i])+1,1), agent_loss[i], label = "Agent %d" %(i+1))

    # plt.figure(1)
    # plt.plot(np.arange(1,len(training_accuracy[i])+1,1), training_accuracy[i], label = "Agent %d" %(i+1))

    plt.figure(2)
    plt.plot(np.arange(1,40*len(testing_accuracy[i])+1,40), testing_accuracy[i], label = "Agent %d" %(i+1))

# for figure in range(3):
#     plt.figure(figure)
#     plt.title("Error")
#     plt.xlabel("Epochs", fontsize=18)
#     plt.ylabel("Accuracy %", fontsize=18)
#     plt.legend(fontsize=18)

# plt.figure(0)
# plt.title("Loss Function Accuracy", fontsize=18)
# plt.xlabel("Epochs", fontsize=18)
# plt.ylabel("Accuracy %", fontsize=18)
# plt.legend(fontsize=18)

# plt.figure(1)
# plt.title("Training Accuracy", fontsize=18)
# plt.xlabel("Epochs", fontsize=18)
# plt.ylabel("Accuracy %", fontsize=18)
# plt.legend(fontsize=18)

plt.figure(2)
plt.xlabel("Epochs", fontsize=18)
plt.ylabel("Accuracy %", fontsize=18)
plt.legend(fontsize=18)

# plt.figure(3)
# plt.plot(np.arange(1,10*len(control_batch_size_accuracy)+1,10), control_batch_size_accuracy, label = "Centralized")
# for i in range(5):
#     plt.plot(np.arange(1,10*len(batch_size_accuracy[i])+1,10), batch_size_accuracy[i], label = "Agent %d" %(i+1))

# plt.title("Test Accuracy")
# plt.xlabel("Number of Communication Rounds")
# plt.ylabel("Accuracy %")
# plt.legend()
plt.show()