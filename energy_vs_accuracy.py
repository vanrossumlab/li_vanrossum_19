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
    train_data = np.loadtxt('../../mnist_train_100.csv',delimiter=',')
    fac = 255 # normalising data values to [0., 1.]
    train_imgs = np.asfarray(train_data[:, 1:]) / fac
    train_labels = np.asfarray(train_data[:, :1])
else:
    train_imgs = np.fromfile('../../mnist_train_imgs_binary.dat').reshape((60000,image_pixels))
    train_labels = np.fromfile('../../mnist_train_labels_binary.dat').reshape((60000,1))
test_imgs = np.fromfile('../../mnist_test_imgs_binary.dat').reshape((10000,image_pixels))
test_labels = np.fromfile('../../mnist_test_labels_binary.dat').reshape((10000,1))


no_of_hidden_nodes = 100
accuracy = 0.97
epochs = 30
bias = None
nAlgorithm = 3 # excluding minibatch
run_minibatch = False
run_Adam_minibatch = False
run_cross_entropy_error = False
learning_rate = 0.1
threshold = [0,0.02,0.04]#0.08]
size_minibatch = 400
size_minibatch_Adam = 20
accuracy_buffer = 3 #when to stop the code if accuracy not improving
save_frequency = 1000 #energy & accuracy are measured after running this amount
decay_eLTP = 0.001
energy_scale_maintenance = 0.001

network = Klein.NeuralNetwork(no_of_in_nodes=image_pixels,
                              no_of_out_nodes=no_of_different_labels,
                              no_of_hidden_nodes=no_of_hidden_nodes,
                              learning_rate=learning_rate,
                              data_array=train_imgs,
                              labels=train_labels,
                              no_of_different_labels=no_of_different_labels,
                              accuracy=accuracy,
                              epochs=epochs,
                              bias=bias,
                              mean_square_error_cost=True,
                              intermediate_results=True,
                              data_array_for_testing=test_imgs,
                              labels_for_testing=test_labels)


bias_node = 1 if bias else 0
function = ['AlgoStandard','AlgoSynapse','AlgoLocalThresGlobalCon']#'AlgoAdamQ']
wih = [] # weights between input and hidden layers
who = [] # weights between hidden and output layers
energy = []
accuracy = [] # save the accuracy rate
# run algorithms without minibatch

network.decay_eLTP = 0.
network.energy_scale_maintenance = 0.
# find the theoretical minimum
# tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_more_save('AlgoSynapse', accuracy_buffer, save_frequency, threshold=999999999)
# wih.append(tmp_wih[-1])
# who.append(tmp_who[-1])
# energy.append(tmp_e)
# accuracy.append(tmp_accu)
# np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo0.txt',wih[0]) 
# np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo0.txt',wih[0])
# np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo0.txt',who[0])
# np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo0.txt',accuracy[0])
# np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo0.txt',energy[0])

# train with various learning rules with distinct decay constant and epilson
network.decay_eLTP = decay_eLTP
network.energy_scale_maintenance = energy_scale_maintenance

for iAlgo in range(1,nAlgorithm+1):
    if(iAlgo!=nAlgorithm): continue
    print('Train ** algorithm ', iAlgo)
    tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_more_save(function[iAlgo-1], accuracy_buffer, save_frequency, threshold=threshold[iAlgo-1])
    wih.append(tmp_wih[-1])
    who.append(tmp_who[-1])
    energy.append(tmp_e)
    accuracy.append(tmp_accu)
    print(energy[iAlgo])

    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(iAlgo)+'.txt',accuracy[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(iAlgo)+'.txt',energy[iAlgo])

        
# run minibatch
if(run_minibatch is True):
    iAlgo = iAlgo+1
    print('Train ** algorithm ', iAlgo, ' , learning rate:',learning_rate)
    network.learning_rate = learning_rate
    tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_minibatch_more_save('AlgoStandard', size_minibatch, accuracy_buffer, save_frequency)
    wih.append(tmp_wih[-1])
    who.append(tmp_who[-1])
    energy.append(tmp_e)
    accuracy.append(tmp_accu)
    print(energy[iAlgo])

    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(iAlgo)+'.txt',accuracy[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(iAlgo)+'.txt',energy[iAlgo])
    
# run Adam with minibatch
if(run_Adam_minibatch is True):
    iAlgo = iAlgo+1
    network.learning_rate = learning_rate
    tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_minibatch_more_save('AlgoAdamQ', size_minibatch_Adam, accuracy_buffer, save_frequency)
    wih.append(tmp_wih[-1])
    who.append(tmp_who[-1])
    energy.append(tmp_e)
    accuracy.append(tmp_accu)
    print(energy[iAlgo])

    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(iAlgo)+'.txt',accuracy[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(iAlgo)+'.txt',energy[iAlgo])
    


# run cross entropy error function
if(run_cross_entropy_error):
    network.mean_square_error_cost = False
    network.decay_eLTP = 0.
    network.energy_scale_maintenance = 0.
    iAlgo = iAlgo+1
    # find the theoretical minimum
    tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_more_save('AlgoSynapse', accuracy_buffer, save_frequency, threshold=999999999)
    wih.append(tmp_wih[-1])
    who.append(tmp_who[-1])
    energy.append(tmp_e)
    accuracy.append(tmp_accu)
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(iAlgo)+'.txt',accuracy[iAlgo])
    np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(iAlgo)+'.txt',energy[iAlgo])

    # train with various learning rules with distinct decay constant and epilson
    network.decay_eLTP = decay_eLTP
    network.energy_scale_maintenance = energy_scale_maintenance

    for tmp_iAlgo in range(iAlgo+1,nAlgorithm+iAlgo+1):
        print('Train ** algorithm ', tmp_iAlgo)
        tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_more_save(function[tmp_iAlgo-iAlgo-1], accuracy_buffer, save_frequency, threshold=threshold[tmp_iAlgo-iAlgo-1])
        wih.append(tmp_wih[-1])
        who.append(tmp_who[-1])
        energy.append(tmp_e)
        accuracy.append(tmp_accu)
        print(energy[tmp_iAlgo])

        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(tmp_iAlgo)+'.txt',accuracy[tmp_iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(tmp_iAlgo)+'.txt',energy[tmp_iAlgo])
    
        
    # run minibatch
    if(run_minibatch is True):
        iAlgo = tmp_iAlgo+1
        print('Train ** algorithm ', iAlgo, ' , learning rate:',learning_rate)
        network.learning_rate = learning_rate
        tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_minibatch_more_save('AlgoStandard', size_minibatch, accuracy_buffer, save_frequency)
        wih.append(tmp_wih[-1])
        who.append(tmp_who[-1])
        energy.append(tmp_e)
        accuracy.append(tmp_accu)
        print(energy[iAlgo])

        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])        
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(iAlgo)+'.txt',accuracy[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(iAlgo)+'.txt',energy[iAlgo])

    # run Adam with minibatch
    if(run_Adam_minibatch is True):
        iAlgo = iAlgo+1
        network.learning_rate = learning_rate
        tmp_wih, tmp_who, tmp_e, tmp_accu = network.train_minibatch_more_save('AlgoAdamQ', size_minibatch_Adam, accuracy_buffer, save_frequency)
        wih.append(tmp_wih[-1])
        who.append(tmp_who[-1])
        energy.append(tmp_e)
        accuracy.append(tmp_accu)
        print(energy[iAlgo])

        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_wih_algo'+str(iAlgo)+'.txt',wih[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_who_algo'+str(iAlgo)+'.txt',who[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo'+str(iAlgo)+'.txt',accuracy[iAlgo])
        np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo'+str(iAlgo)+'.txt',energy[iAlgo])

    
#np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_other_variable.txt',(no_of_hidden_nodes,learning_rate,threshold,size_minibatch,size_minibatch_Adam,decay_eLTP,energy_scale_maintenance))
np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_other_variable.txt',(no_of_hidden_nodes,learning_rate,size_minibatch,size_minibatch_Adam,decay_eLTP,energy_scale_maintenance))
np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_threshold.txt',threshold)
np.savetxt('Text/energy_vs_accuracy_20190525/energy_vs_accuracy_run_option.txt',(run_minibatch,run_Adam_minibatch,run_cross_entropy_error))

