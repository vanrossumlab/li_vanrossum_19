#!/usr/bin/env python
import numpy as np
#@np.vectorize

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
activation_function = sigmoid

from scipy.stats import truncnorm
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)

def transform_labels_into_one_hot(labels, no_of_different_labels):
    lr = np.arange(no_of_different_labels)
    # transform labels into one hot representation
    labels_one_hot = (lr==labels).astype(np.float)
    # we don't want zeroes and ones in the labels neither:
    labels_one_hot[labels_one_hot==0] = 0.01
    labels_one_hot[labels_one_hot==1] = 0.99
    return labels_one_hot
    
class NeuralNetwork:

        def __init__(self,
                     no_of_in_nodes,
                     no_of_out_nodes,
                     no_of_hidden_nodes,
                     learning_rate,
                     data_array,
                     labels,
                     no_of_different_labels,
                     accuracy=0.96,
                     epochs=20,
                     bias=None,
                     mean_square_error_cost=True,
                     intermediate_results=False,
                     data_array_for_testing=None,
                     labels_for_testing=None
                     ):
            self.no_of_in_nodes = no_of_in_nodes
            self.no_of_out_nodes = no_of_out_nodes
            self.no_of_hidden_nodes = no_of_hidden_nodes
            
            self.learning_rate = learning_rate
            self.data_array = data_array
            self.labels = labels
            self.no_of_different_labels = no_of_different_labels
            self.accuracy = accuracy
            self.epochs = epochs
            self.mean_square_error_cost = mean_square_error_cost # True: mean square error; False: cross entropy error
            self.intermediate_results = intermediate_results
            
            self.energy_scale_maintenance = 0.
            self.decay_eLTP = 0.
            self.energy = 0.
            self.energy_eLTP = 0.
            self.energy_lLTP = 0.

            self.create_weight_matrices()
            self.labels_one_hot = transform_labels_into_one_hot(labels, no_of_different_labels)

            if(data_array_for_testing is None):
                self.data_array_for_testing = data_array
                self.labels_for_testing = self.labels_one_hot
            else:
                self.data_array_for_testing = data_array_for_testing
                self.labels_for_testing = transform_labels_into_one_hot(labels_for_testing, no_of_different_labels)

        def map_algorithm(self,algorithm,momentum_para):
            if(algorithm == "AlgoStandard"):
                momentum_para = 0.
                return self.AlgoStandard
            elif(algorithm == "AlgoSynapse"):
                momentum_para = 0.
                return self.AlgoSynapse
            elif(algorithm == "AlgoSynapseQ"):
                momentum_para = 0.
                return self.AlgoSynapseQ
            elif(algorithm == "AlgoLocalThresGlobalCon"):
                momentum_para = 0.
                return self.AlgoLocalThresGlobalCon                
            elif(algorithm == "AlgoLocalThresAllCon"):
                momentum_para = 0.
                return self.AlgoLocalThresAllCon                
            elif(algorithm == "AlgoMomentum"):
                return self.AlgoMomentum

            
        def create_weight_matrices(self):
            """ 
            A method to initialize the weight matrices 
            of the neural network with optional 
            bias nodes"""

            # rad = 1 / np.sqrt(self.no_of_in_nodes + self.bias_node) # radius
            # X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            # self.wih_original = X.rvs((self.no_of_hidden_nodes,
            #                            self.no_of_in_nodes + self.bias_node))
            self.wih_original = np.random.normal(0.,0.01,
                                            (self.no_of_hidden_nodes,
                                             self.no_of_in_nodes))
            self.wih = np.copy(self.wih_original)
            self.dWih = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            self.dWih_previous = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            # rad = 1 / np.sqrt(self.no_of_hidden_nodes + self.bias_node)
            # X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            # self.who_original = X.rvs((self.no_of_out_nodes,
            #                            self.no_of_hidden_nodes + self.bias_node))
            self.who_original = np.random.normal(0.,0.01,
                                        (self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.who = np.copy(self.who_original)
            self.dWho = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.dWho_previous = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))

        def regenerate_weight_matrices(self):
            self.wih = np.copy(self.wih_original)
            self.dWih = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            self.dWih_previous = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
            self.who = np.copy(self.who_original)
            self.dWho = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            self.dWho_previous = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
            
        def update_energy(self, ddWho, ddWih):
            self.energy += np.sum( np.fabs(ddWho) )
            self.energy += np.sum( np.fabs(ddWih) )
            self.energy_lLTP += np.sum( np.fabs(ddWho) )
            self.energy_lLTP += np.sum( np.fabs(ddWih) )
            self.energy += self.energy_scale_maintenance * ( np.sum( np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)) )
            self.energy_eLTP += self.energy_scale_maintenance * ( np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)) )
            self.dWih = self.dWih * np.exp(-self.decay_eLTP)
            self.dWho = self.dWho * np.exp(-self.decay_eLTP)
                
                
        def AlgoStandard(self, threshold):
                self.who += self.dWho
                ddWho = np.copy(self.dWho)
                self.dWho.fill(0.)
                self.wih += self.dWih
                ddWih = np.copy(self.dWih)
                self.dWih.fill(0.)
                
                self.update_energy(ddWho, ddWih)

        def AlgoMomentum(self, threshold):
                self.who += self.dWho
                self.dWho_previous = np.copy(self.dWho)
                self.dWho.fill(0.)
                self.wih += self.dWih
                self.dWih_previous = np.copy(self.dWih)
                self.dWih.fill(0.)
                
                self.update_energy(self.dWho_previous, self.dWih_previous)
                
        def AlgoSynapse(self, threshold):
                ddWho = np.where(np.fabs(self.dWho)>threshold, self.dWho, 0.)
                self.who += ddWho
                self.dWho -= ddWho
                ddWih = np.where(np.fabs(self.dWih)>threshold, self.dWih, 0.)  
                self.wih += ddWih
                self.dWih -= ddWih

                self.update_energy(ddWho, ddWih)

        def AlgoSynapseQ(self, threshold):
                ddWho = threshold*(self.dWho//threshold)
                self.who += ddWho
                self.dWho -= ddWho
                ddWih = threshold*(self.dWih//threshold)
                self.wih += ddWih
                self.dWih -= ddWih
                
                self.update_energy(ddWho, ddWih)

        # local threshold, global consolidation
        def AlgoLocalThresGlobalCon(self, threshold):
                ddWho = np.zeros(shape=(self.no_of_out_nodes,
                                        self.no_of_hidden_nodes))
                ddWih = np.zeros(shape=(self.no_of_hidden_nodes,
                                        self.no_of_in_nodes))
                tmp = np.where(np.fabs(self.dWho)>threshold)
                # if(len(tmp[1])>0):
                #     for iNode in tmp[1]:
                #         ddWho.T[iNode] += self.dWho.T[iNode]
                #         self.dWho.T[iNode].fill(0.)
                if(len(tmp[0])>0):
                    for iNode in tmp[0]:
                        ddWho[iNode] += self.dWho[iNode]
                        self.dWho[iNode].fill(0.)
                    self.who += ddWho
                tmp = np.where(np.fabs(self.dWih)>threshold)
                # if(len(tmp[1])>0):
                #     for iNode in tmp[1]:
                #         ddWih.T[iNode] += self.dWih.T[iNode]
                #         self.dWih.T[iNode].fill(0.)
                if(len(tmp[0])>0):
                    for iNode in tmp[0]:
                        ddWih[iNode] += self.dWih[iNode]
                        self.dWih[iNode].fill(0.)
                    self.wih += ddWih

                self.update_energy(ddWho, ddWih)                    

        # local threshold, all consolidation
        def AlgoLocalThresAllCon(self, threshold):
                tmp = np.where(np.fabs(self.dWho)>threshold)
                if(len(tmp[0])>0):
                    self.who += self.dWho
                    ddWho = np.copy(self.dWho)
                    self.dWho.fill(0.)
                else:
                    ddWho = 0.
                tmp = np.where(np.fabs(self.dWih)>threshold)
                if(len(tmp[0])>0):
                    self.wih += self.dWih
                    ddWih = np.copy(self.dWih)
                    self.dWih.fill(0.)
                else:
                    ddWih = 0.

                self.update_energy(ddWho, ddWih)                    
                
        def train_single(self, input_vector, target_vector,
                         algorithm, threshold, momentum_para):
                """
                input_vector and target_vector can be tuple, 
                list or ndarray
                """
                input_vector = np.array(input_vector, ndmin=2).T

                output_hidden = activation_function(np.dot(self.wih+self.dWih,input_vector))
                    
                output_network = activation_function(np.dot(self.who+self.dWho, output_hidden))

                # actually E=(target-output)^2, so outputerror = -dE/dy
                output_errors = np.array(target_vector, ndmin=2).T - output_network
                if self.mean_square_error_cost:
                    # find update to hid->out weights:
                    tmp = output_errors * output_network * (1.0 - output_network)
                    # calculate hidden errors:
                    hidden_errors = np.dot(self.who.T+self.dWho.T, output_errors)
                    # find update to the in -> hid weights:
                    tmp2 = hidden_errors * output_hidden * (1.0 - output_hidden)
                else:
                    tmp = output_errors
                    hidden_errors = np.dot(self.who.T+self.dWho.T, output_errors)
                    tmp2 = hidden_errors * output_hidden
    
                y = np.dot(tmp, output_hidden.T)
                
                x = np.dot(tmp2, input_vector.T)


                self.dWho += self.learning_rate * y + momentum_para * self.dWho_previous
                self.dWih += self.learning_rate * x + momentum_para * self.dWih_previous
                
                algorithm(threshold) 


#        @profile                
        def train_single_minibatch(self, input_vector, target_vector, momentum_para):
                """
                input_vector and target_vector can be tuple, 
                list or ndarray
                """
                input_vector = np.array(input_vector, ndmin=2).T

                output_hidden = activation_function(np.dot(self.wih+self.dWih,input_vector))
                    
                output_network = activation_function(np.dot(self.who+self.dWho, output_hidden))

                # actually E=(target-output)^2, so outputerror = -dE/dy
                output_errors = np.array(target_vector, ndmin=2).T - output_network
                if self.mean_square_error_cost:
                    # find update to hid->out weights:
                    tmp = output_errors * output_network * (1.0 - output_network)
                    # calculate hidden errors:
                    hidden_errors = np.dot(self.who.T+self.dWho.T, output_errors)
                    # find update to the in -> hid weights:
                    tmp2 = hidden_errors * output_hidden * (1.0 - output_hidden)
                else:
                    tmp = output_errors
                    hidden_errors = np.dot(self.who.T+self.dWho.T, output_errors)
                    tmp2 = hidden_errors * output_hidden

                y = np.dot(tmp, output_hidden.T)
                x = np.dot(tmp2, input_vector.T)

                self.dWho += self.learning_rate * y + momentum_para * self.dWho_previous
                self.dWih += self.learning_rate * x + momentum_para * self.dWih_previous
                
                    
        def train_minibatch(self, algorithm, size_minibatch, accuracy_buffer,
                            threshold=0., momentum_para=0.,
                            energy_detail=False):
            # accuracy_buffer: when to stop the code if accuracy not improving
            # energy_detail=True would give eLTP and lLTP energies individually
            intermediate_wih = []
            intermediate_who = []
            energy = []
            energy_eLTP = []
            energy_lLTP = []
            accuracy = []
            algorithm = self.map_algorithm(algorithm,0.)
            
            self.energy = 0.; self.energy_eLTP = 0.; self.energy_lLTP = 0.
            self.regenerate_weight_matrices()
            iPattern_run = 0
            for epoch in range(self.epochs):
                corrects, wrongs = 0, 0
                for i in range(len(self.data_array)):
                    self.train_single_minibatch(self.data_array[i],
                                                self.labels_one_hot[i],
                                                momentum_para)
                    if(((iPattern_run+1)%size_minibatch)==0):
                        algorithm(threshold)
                    else:
                        self.energy += self.energy_scale_maintenance * ( np.sum( np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)) )
                        self.energy_eLTP += self.energy_scale_maintenance * ( np.sum( np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)) )
                        self.dWih = self.dWih * np.exp(-self.decay_eLTP)
                        self.dWho = self.dWho * np.exp(-self.decay_eLTP)
                        
                    iPattern_run += 1

                if self.intermediate_results:
                    intermediate_wih.append(self.wih.copy())
                    intermediate_who.append(self.who.copy())
                    # find E cost if all eLTP is converted to lLTP          
                    energy.append(self.energy + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                    energy_eLTP.append(self.energy_eLTP)
                    energy_lLTP.append(self.energy_lLTP + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                    energy_lLTP.append(self.energy_lLTP)
                for i in range(len(self.data_array_for_testing)):
                    res = self.run(self.data_array_for_testing[i])
                    res_max = res.argmax()
                    if res_max == self.labels_for_testing[i].argmax():
                        corrects += 1
                    else:
                        wrongs += 1
                accuracy.append(1.0*corrects / ( corrects + wrongs))
                print("accuracy: ", accuracy[epoch])
                if(accuracy[epoch] > self.accuracy):
                    break
                if( (len(accuracy)>(accuracy_buffer+1)) and (accuracy[epoch] <= min(accuracy[-(accuracy_buffer+1):-1])) ):
                    break
                
            if(energy_detail is False):
                return intermediate_wih, intermediate_who, energy, accuracy
            else:
                return intermediate_wih, intermediate_who, energy, energy_eLTP, energy_lLTP, accuracy

        def train_minibatch_more_save(self, algorithm, size_minibatch,
                                      accuracy_buffer, save_frequency,
                                      threshold=0., momentum_para=0.,
                                      energy_detail=False):
            # accuracy_buffer: when to stop the code if accuracy not improving
            # save_frequency: how often energy & accuracy are measured
            # energy_detail=True would give eLTP and lLTP energies individually
            intermediate_wih = []
            intermediate_who = []
            energy = []
            energy_eLTP = []
            energy_lLTP = []
            accuracy = []
            accuracy_more_save = []
            algorithm = self.map_algorithm(algorithm,0.)
            
            self.energy = 0.; self.energy_eLTP = 0.; self.energy_lLTP = 0.
            self.regenerate_weight_matrices()
            iPattern_run = 0
            for epoch in range(self.epochs):
                for i in range(len(self.data_array)):
                    self.train_single_minibatch(self.data_array[i],
                                                self.labels_one_hot[i],
                                                momentum_para)
                    self.energy += self.energy_scale_maintenance * ( np.sum( np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)) )
                    self.energy_eLTP += self.energy_scale_maintenance * ( np.sum( np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)) )
                    self.dWih = self.dWih * np.exp(-self.decay_eLTP)
                    self.dWho = self.dWho * np.exp(-self.decay_eLTP)

                    if(((iPattern_run+1)%size_minibatch)==0):
                        algorithm(threshold)
                    iPattern_run += 1
                        
                    if(((i+1)%save_frequency)==0):
                        corrects, wrongs = 0, 0
                        intermediate_wih.append(self.wih.copy())
                        intermediate_who.append(self.who.copy())
                        # find E cost if all eLTP is converted to lLTP
                        energy.append(self.energy + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                        energy_eLTP.append(self.energy_eLTP)
                        energy_lLTP.append(self.energy_lLTP + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                        for idata in range(len(self.data_array_for_testing)):
                            res = self.run(self.data_array_for_testing[idata])
                            res_max = res.argmax()
                            if res_max == self.labels_for_testing[idata].argmax():
                                corrects += 1
                            else:
                                wrongs += 1
                        accuracy_more_save.append(1.0*corrects / ( corrects + wrongs))
                        print(accuracy_more_save[-1])
                accuracy.append(accuracy_more_save[-1])
                print("accuracy: ", accuracy[epoch])
                if(accuracy[epoch] > self.accuracy):
                    break
                if( (len(accuracy)>(accuracy_buffer+1)) and (accuracy[epoch] <= min(accuracy[-(accuracy_buffer+1):-1])) ):
                    break
                
            if(energy_detail is False):
                return intermediate_wih, intermediate_who, energy, accuracy_more_save
            else:
                return intermediate_wih, intermediate_who, energy, energy_eLTP, energy_lLTP, accuracy_more_save

        
        def train(self, algorithm, accuracy_buffer,
                  threshold=0., momentum_para=0., energy_detail=False):
            # accuracy_buffer: when to stop the code if accuracy not improving
            # energy_detail=True would give eLTP and lLTP energies individually
            intermediate_wih = []
            intermediate_who = []
            energy = []
            energy_eLTP = []
            energy_lLTP = []
            accuracy = []
            algorithm = self.map_algorithm(algorithm,momentum_para)
            
            self.energy = 0.; self.energy_eLTP = 0.; self.energy_lLTP = 0.
            self.regenerate_weight_matrices()
            for epoch in range(self.epochs):
                corrects, wrongs = 0, 0
                for i in range(len(self.data_array)):
                    self.train_single(self.data_array[i],
                                      self.labels_one_hot[i],
                                      algorithm, threshold,
                                      momentum_para)
                if self.intermediate_results:
                    intermediate_wih.append(self.wih.copy())
                    intermediate_who.append(self.who.copy())
                    # find E cost if all eLTP is converted to lLTP          
                    energy.append(self.energy + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                    energy_eLTP.append(self.energy_eLTP)
                    energy_lLTP.append(self.energy_lLTP + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                for i in range(len(self.data_array_for_testing)):
                    res = self.run(self.data_array_for_testing[i])
                    res_max = res.argmax()
                    if res_max == self.labels_for_testing[i].argmax():
                        corrects += 1
                    else:
                        wrongs += 1
                accuracy.append(1.0*corrects / ( corrects + wrongs))
                print("accuracy: ", accuracy[epoch])
                if(accuracy[epoch] > self.accuracy):
                    break
                if( (len(accuracy)>(accuracy_buffer+1)) and (accuracy[epoch] <= min(accuracy[-(accuracy_buffer+1):-1])) ):
                    break
                    
            if(energy_detail is False):
                return intermediate_wih, intermediate_who, energy, accuracy
            else:
                return intermediate_wih, intermediate_who, energy, energy_eLTP, energy_lLTP, accuracy

        
        def train_more_save(self, algorithm, accuracy_buffer, save_frequency,
                            threshold=0., momentum_para=0.,
                            energy_detail=False):
            # accuracy_buffer: when to stop the code if accuracy not improving
            # save_frequency: how often energy & accuracy are measured
            # energy_detail=True would give eLTP and lLTP energies individually
            intermediate_wih = []
            intermediate_who = []
            energy = []
            energy_eLTP = []
            energy_lLTP = []
            accuracy = []
            accuracy_more_save = []
            algorithm = self.map_algorithm(algorithm,momentum_para)
            
            self.energy = 0.; self.energy_eLTP = 0.; self.energy_lLTP = 0.
            self.regenerate_weight_matrices()
            for epoch in range(self.epochs):
                for i in range(len(self.data_array)):
                    self.train_single(self.data_array[i],
                                      self.labels_one_hot[i],
                                      algorithm, threshold,
                                      momentum_para)
                    if(((i+1)%save_frequency)==0):
                        corrects, wrongs = 0, 0
                        intermediate_wih.append(self.wih.copy())
                        intermediate_who.append(self.who.copy())
                        # find E cost if all eLTP is converted to lLTP
                        energy.append(self.energy + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                        energy_eLTP.append(self.energy_eLTP)
                        energy_lLTP.append(self.energy_lLTP + np.sum(np.fabs(self.dWih)) + np.sum(np.fabs(self.dWho)))
                        for idata in range(len(self.data_array_for_testing)):
                            res = self.run(self.data_array_for_testing[idata])
                            res_max = res.argmax()
                            if res_max == self.labels_for_testing[idata].argmax():
                                corrects += 1
                            else:
                                wrongs += 1
                        accuracy_more_save.append(1.0*corrects / ( corrects + wrongs))
                accuracy.append(accuracy_more_save[-1])
                print("accuracy: ", accuracy[epoch])
                if(accuracy[epoch] > self.accuracy):
                    break
                if( (len(accuracy)>(accuracy_buffer+1)) and (accuracy[epoch] <= min(accuracy[-(accuracy_buffer+1):-1])) ):
                    break

            if(energy_detail is False):
                return intermediate_wih, intermediate_who, energy, accuracy_more_save
            else:
                return intermediate_wih, intermediate_who, energy, energy_eLTP, energy_lLTP, accuracy_more_save

        
        def run(self, input_vector):
            # input_vector can be tuple, list or ndarray
            input_vector = np.array(input_vector, ndmin=2).T
            output_vector = np.dot(self.wih+self.dWih,
                                   input_vector)
            output_vector = activation_function(output_vector)
            
            output_vector = np.dot(self.who+self.dWho,
                                   output_vector)
            output_vector = activation_function(output_vector)
                        
            return output_vector
        

        def evaluate(self, data, labels):
            corrects, wrongs = 0, 0
            for i in range(len(data)):
                res = self.run(data[i])
                res_max = res.argmax()
                if res_max == labels[i]:
                    corrects += 1
                else:
                    wrongs += 1
            return corrects, wrongs
