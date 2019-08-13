#!/usr/bin/env python
# from https://www.python-course.eu/neural_network_mnist.php
import numpy as np
import matplotlib.pyplot as plt
import KleinFunction as Klein

image_size = 28
no_of_different_labels = 10
image_pixels = image_size*image_size
usebinQ = True
if(usebinQ is False):
    #train_data = np.loadtxt("../../mnist_train.csv",delimiter=",")
    train_data = np.loadtxt("../../mnist_train_100.csv",delimiter=",")
    fac = 255 # normalising data values to [0., 1.]
    train_imgs = np.asfarray(train_data[:, 1:]) / fac
    train_labels = np.asfarray(train_data[:, :1])
else:
    train_imgs = np.fromfile("../../mnist_train_imgs_binary.dat").reshape((60000,image_pixels))
    train_labels = np.fromfile("../../mnist_train_labels_binary.dat").reshape((60000,1))
test_imgs = np.fromfile("../../mnist_test_imgs_binary.dat").reshape((10000,image_pixels))
test_labels = np.fromfile("../../mnist_test_labels_binary.dat").reshape((10000,1))


no_of_hidden_nodes = 100
accuracy = 0.96
epochs = 100#30
bias = None
nAlgorithm = 1 # excluding minibatch
run_minibatch = False
arr_learning_rate = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0]
save_frequency = 6000
threshold = 0.0005
arr_momentum_para = [0.,0.9,0.] # 0.9 is for the algorithm with momentum
size_minibatch = 20
accuracy_buffer = 3
network = Klein.NeuralNetwork(no_of_in_nodes=image_pixels,
                              no_of_out_nodes=no_of_different_labels,
                              no_of_hidden_nodes=no_of_hidden_nodes,
                              learning_rate=arr_learning_rate[0],
                              data_array=train_imgs,
                              labels=train_labels,
                              no_of_different_labels=no_of_different_labels,
                              accuracy=accuracy,
                              epochs=epochs,
                              bias=bias,
                              intermediate_results=True,
                              data_array_for_testing=test_imgs,
                              labels_for_testing=test_labels)

bias_node = 1 if bias else 0
function = ["AlgoStandard","AlgoMomentum","AlgoSynapse"]
energy = []
energy_eLTP = []
energy_lLTP = []
energy_total = []
energy_eLTP_total = []
energy_lLTP_total = []
arr_accuracy = []
arr_accuracy_final = []
# run algorithms without minibatch
for iAlgo in range(nAlgorithm):
    energy.append([])
    energy_eLTP.append([])
    energy_lLTP.append([])
    energy_total.append([])
    energy_eLTP_total.append([])
    energy_lLTP_total.append([])
    arr_accuracy.append([])
    arr_accuracy_final.append([])
    # train
    for iRate in range(len(arr_learning_rate)):
        print("Train ** algorithm ", iAlgo, " , learning rate:",arr_learning_rate[iRate])
        network.learning_rate = arr_learning_rate[iRate]
        tmp_wih, tmp_who, tmp_e, tmp_e_e, tmp_e_l, tmp_accu = network.train_more_save(function[iAlgo], accuracy_buffer, save_frequency, threshold=threshold, momentum_para=arr_momentum_para[iAlgo], energy_detail=True)
        energy[iAlgo].append(tmp_e)
        energy_eLTP[iAlgo].append(tmp_e_e)
        energy_lLTP[iAlgo].append(tmp_e_l)
        energy_total[iAlgo].append(tmp_e[-1])
        energy_eLTP_total[iAlgo].append(tmp_e_e[-1])
        energy_lLTP_total[iAlgo].append(tmp_e_l[-1])
        arr_accuracy[iAlgo].append(tmp_accu)
        arr_accuracy_final[iAlgo].append(tmp_accu[-1])
        print(energy[iAlgo][iRate])

        np.savetxt("Text/learning_rate_accuracy_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",arr_accuracy[iAlgo][iRate])
        np.savetxt("Text/learning_rate_energy_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",energy[iAlgo][iRate])
        np.savetxt("Text/learning_rate_energy_eLTP_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",energy_eLTP[iAlgo][iRate])
        np.savetxt("Text/learning_rate_energy_lLTP_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",energy_lLTP[iAlgo][iRate])

        
# run minibatch
if(run_minibatch is True):
    iAlgo = nAlgorithm
    energy.append([])
    energy_eLTP.append([])
    energy_lLTP.append([])
    energy_total.append([])
    energy_eLTP_total.append([])
    energy_lLTP_total.append([])
    arr_accuracy.append([])
    arr_accuracy_final.append([])
    # train
    for iRate in range(len(arr_learning_rate)):
        print("Train ** algorithm ", iAlgo, " , learning rate:",arr_learning_rate[iRate])
        network.learning_rate = arr_learning_rate[iRate]
        tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_minibatch_more_save("AlgoStandard", size_minibatch, accuracy_buffer, save_frequency, energy_detail=True)
        energy[iAlgo].append(tmp_e)
        energy_eLTP[iAlgo].append(tmp_e_e)
        energy_lLTP[iAlgo].append(tmp_e_l)
        energy_total[iAlgo].append(tmp_e[-1])
        energy_eLTP_total[iAlgo].append(tmp_e_e[-1])
        energy_lLTP_total[iAlgo].append(tmp_e_l[-1])
        arr_accuracy[iAlgo].append(tmp_accu)
        arr_accuracy_final[iAlgo].append(tmp_accu[-1])
        print(energy[iAlgo][iRate])

        np.savetxt("Text/learning_rate_accuracy_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",arr_accuracy[iAlgo][iRate])
        np.savetxt("Text/learning_rate_energy_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",energy[iAlgo][iRate])
        np.savetxt("Text/learning_rate_energy_eLTP_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",energy_eLTP[iAlgo][iRate])
        np.savetxt("Text/learning_rate_energy_lLTP_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt",energy_lLTP[iAlgo][iRate])
    

np.savetxt("Text/learning_rate_accuracy_final.txt",arr_accuracy_final)
np.savetxt("Text/learning_rate_energy_total.txt",energy_total)
np.savetxt("Text/learning_rate_energy_eLTP_total.txt",energy_eLTP_total)
np.savetxt("Text/learning_rate_energy_lLTP_total.txt",energy_lLTP_total)
np.savetxt("Text/learning_rate_learning_rate.txt",arr_learning_rate)
np.savetxt("Text/learning_rate_momentum_para.txt",arr_momentum_para)
np.savetxt("Text/learning_rate_other_variable.txt",(threshold,size_minibatch,save_frequency))
