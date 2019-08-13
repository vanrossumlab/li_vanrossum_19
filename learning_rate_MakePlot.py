#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np

arr_learning_rate = np.loadtxt("Text/forMark/learning_rate_learning_rate.txt")

nAlgorithm = 1
arr_accuracy = []
energy = []
for iAlgo in range(nAlgorithm):
    arr_accuracy.append([])
    energy.append([])
    for iRate in range(len(arr_learning_rate)):
        arr_accuracy[iAlgo].append(np.loadtxt("Text/forMark/learning_rate_accuracy_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt"))
        energy[iAlgo].append(np.loadtxt("Text/forMark/learning_rate_energy_algo"+str(iAlgo)+"_rate"+str(arr_learning_rate[iRate])+".txt"))


arr_color = ["black","gray","firebrick","red","darkorange","gold","lawngreen","green","royalblue","blue","violet","purple"]
arr_linestyle=["-","--","-.",":"]

for iAlgo in range(nAlgorithm):
    for iRate in range(len(arr_learning_rate)):
        #if(iRate!=2): continue
        plt.plot(arr_accuracy[iAlgo][iRate],energy[iAlgo][iRate],color=arr_color[iRate],linestyle=arr_linestyle[iAlgo],label=arr_learning_rate[iRate])
plt.xlim(0.7,0.97)
plt.ylim(0.0,5e5)
plt.xlabel("accuracy",fontsize=16)
plt.ylabel("energy [arbitrary unit]",fontsize=16)
plt.legend(bbox_to_anchor=(0.,1.02,1.,0.112),fontsize=10,loc="lower left",ncol=2,mode="expand",borderaxespad=0.)
plt.tight_layout()
plt.savefig("Plot/learning_rate_study_energy_vs_accuracy.png",bbox_inches="tight"); plt.close()

