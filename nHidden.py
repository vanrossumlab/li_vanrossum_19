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

no_of_hidden_nodes = np.linspace(155,5,31) #np.linspace(200,5,40) #np.linspace(160,10,31)
accuracy_required = [0.85]#[0.85,0.93]
epochs = 30
learning_rate = 0.1
accuracy_buffer = 3
save_frequency = 1000
decay_eLTP = 0.001
energy_scale_maintenance = 0.001

standard_energy = []
standard_energy_eLTP = []
standard_energy_lLTP = []
standard_accuracy = []
caching_energy = []
caching_energy_eLTP = []
caching_energy_lLTP = []
caching_accuracy = []
nHidden_performed = []
arr_threshold = []
for iAccu in range(len(accuracy_required)):
    standard_energy.append([])
    standard_energy_eLTP.append([])
    standard_energy_lLTP.append([])
    standard_accuracy.append([])
    caching_energy.append([])
    caching_energy_eLTP.append([])
    caching_energy_lLTP.append([])
    caching_accuracy.append([])
    nHidden_performed.append([])
    arr_threshold.append([])
    for iNode in range(len(no_of_hidden_nodes)):
        network = Klein.NeuralNetwork(no_of_in_nodes=image_pixels,
                                      no_of_out_nodes=no_of_different_labels,
                            no_of_hidden_nodes=int(no_of_hidden_nodes[iNode]),
                                      learning_rate=learning_rate,
                                      data_array=train_imgs,
                                      labels=train_labels,
                                no_of_different_labels=no_of_different_labels,
                                      accuracy=accuracy_required[iAccu],
                                      epochs=epochs,
                                      bias=None,
                                      mean_square_error_cost=True,
                                      intermediate_results=True,
                                      data_array_for_testing=test_imgs,
                                      labels_for_testing=test_labels)
        network.decay_eLTP = decay_eLTP
        network.energy_scale_maintenance = energy_scale_maintenance
        # tmp_wih, tmp_who, tmp_e, tmp_e_e, tmp_e_l, tmp_accu = network.train_more_save('AlgoStandard',accuracy_buffer,save_frequency,energy_detail=True)
        # tmp = np.where( np.array(tmp_accu) > accuracy_required[iAccu] )[0]
        # if(len(tmp)==0):
        #     break
        # nHidden_performed[iAccu].append(no_of_hidden_nodes[iNode])
        # # save the energy when the network first reach the designated accuracy
        # standard_energy[iAccu].append(tmp_e[tmp[0]])
        # standard_energy_eLTP[iAccu].append(tmp_e_e[tmp[0]])
        # standard_energy_lLTP[iAccu].append(tmp_e_l[tmp[0]])
        # standard_accuracy[iAccu].append(tmp_accu[tmp[0]])

        stop_loop = False
        tmp = no_of_hidden_nodes[iNode]
        threshold = 0.01
        caching_energy[iAccu].append([])
        caching_energy_eLTP[iAccu].append([])
        caching_energy_lLTP[iAccu].append([])
        caching_accuracy[iAccu].append([])
        arr_threshold[iAccu].append([])
        while(not stop_loop):
            tmp_wih, tmp_who, tmp_e, tmp_e_e, tmp_e_l, tmp_accu = network.train_more_save('AlgoLocalThresGlobalCon',accuracy_buffer,save_frequency,threshold=threshold,energy_detail=True)
            tmp = np.where( np.array(tmp_accu) > accuracy_required[iAccu] )[0]
            if(len(tmp)==0):
                stop_loop = True
            else:
                caching_energy[iAccu][iNode].append(tmp_e[tmp[0]])
                caching_energy_eLTP[iAccu][iNode].append(tmp_e_e[tmp[0]])
                caching_energy_lLTP[iAccu][iNode].append(tmp_e_l[tmp[0]])
                caching_accuracy[iAccu][iNode].append(tmp_accu[tmp[0]])
                arr_threshold[iAccu][iNode].append(threshold)
                if(len(caching_energy[iAccu][iNode])>2):
                    if(tmp_e[tmp[0]] > caching_energy[iAccu][iNode][-2]):
                        if(tmp_e[tmp[0]] > caching_energy[iAccu][iNode][-3]):
                            stop_loop = True
            threshold += 0.01
        np.savetxt("Text/nHidden_new_3/nHidden_accuracy_AlgoCaching_accuracy"+str(accuracy_required[iAccu])+"_nHidden"+str(int(no_of_hidden_nodes[iNode]))+".txt",caching_accuracy[iAccu][iNode])
        np.savetxt("Text/nHidden_new_3/nHidden_energy_AlgoCaching_accuracy"+str(accuracy_required[iAccu])+"_nHidden"+str(int(no_of_hidden_nodes[iNode]))+".txt",caching_energy[iAccu][iNode])
        np.savetxt("Text/nHidden_new_3/nHidden_energy_eLTP_AlgoCaching_accuracy"+str(accuracy_required[iAccu])+"_nHidden"+str(int(no_of_hidden_nodes[iNode]))+".txt",caching_energy_eLTP[iAccu][iNode])
        np.savetxt("Text/nHidden_new_3/nHidden_energy_lLTP_AlgoCaching_accuracy"+str(accuracy_required[iAccu])+"_nHidden"+str(int(no_of_hidden_nodes[iNode]))+".txt",caching_energy_lLTP[iAccu][iNode])
        np.savetxt("Text/nHidden_new_3/nHidden_threshold_AlgoCaching_accuracy"+str(accuracy_required[iAccu])+"_nHidden"+str(int(no_of_hidden_nodes[iNode]))+".txt",arr_threshold[iAccu][iNode])
        print(no_of_hidden_nodes[iNode]," ",threshold)
        del network
        
    # np.savetxt("Text/nHidden_new_3/nHidden_nHidden_AlgoStandard_accuracy"+str(accuracy_required[iAccu])+".txt",nHidden_performed[iAccu])
    # np.savetxt("Text/nHidden_new_3/nHidden_accuracy_AlgoStandard_accuracy"+str(accuracy_required[iAccu])+".txt",standard_accuracy[iAccu])
    # np.savetxt("Text/nHidden_new_3/nHidden_energy_AlgoStandard_accuracy"+str(accuracy_required[iAccu])+".txt",standard_energy[iAccu])
    # np.savetxt("Text/nHidden_new_3/nHidden_energy_eLTP_AlgoStandard_accuracy"+str(accuracy_required[iAccu])+".txt",standard_energy_eLTP[iAccu])
    # np.savetxt("Text/nHidden_new_3/nHidden_energy_lLTP_AlgoStandard_accuracy"+str(accuracy_required[iAccu])+".txt",standard_energy_lLTP[iAccu])

np.savetxt("Text/nHidden_new_3/nHidden_no_of_hidden_nodes.txt",no_of_hidden_nodes)
np.savetxt("Text/nHidden_new_3/nHidden_other_variable.txt",(learning_rate,save_frequency,decay_eLTP,energy_scale_maintenance))
np.savetxt("Text/nHidden_new_3/nHidden_accuracy_required.txt",accuracy_required)
