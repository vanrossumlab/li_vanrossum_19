#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import Subplot

no_overtrain = True

nAlgorithm = 4
accuracy_required = np.linspace(0.82,0.97,6)
no_of_hidden_nodes, learning_rate, size_minibatch, size_minibatch_Adam, decay_eLTP, energy_scale_maintenance = np.loadtxt("Text/energy_vs_accuracy_20190525/energy_vs_accuracy_other_variable.txt")
threshold  = np.loadtxt("Text/energy_vs_accuracy_20190525/energy_vs_accuracy_threshold.txt")

arr_energy = []
arr_accuracy = []
max_accuracy_index = []
for iAlgo in range(nAlgorithm):
    arr_energy.append( np.loadtxt("Text/energy_vs_accuracy_20190525/energy_vs_accuracy_energy_algo"+str(iAlgo)+".txt") )
    arr_accuracy.append( np.loadtxt("Text/energy_vs_accuracy_20190525/energy_vs_accuracy_accuracy_algo"+str(iAlgo)+".txt") )
    max_accuracy_index.append( np.where(arr_accuracy[-1]==np.max(arr_accuracy[-1]))[0][0] )
    
arr_color = ["lawngreen","red","grey","black"]
arr_label = ["Minimum","No caching","Local threshold, local consolidation","Local threshold, global con."]

fig = plt.figure(facecolor="white",figsize=(5,4))
ax = Subplot(fig,111)
fig.add_subplot(ax)
for iAlgo in [1,3,0]:
    if no_overtrain:
        plt.plot(arr_accuracy[iAlgo][:(max_accuracy_index[iAlgo]+1)],arr_energy[iAlgo][:(max_accuracy_index[iAlgo]+1)],color=arr_color[iAlgo],label=arr_label[iAlgo])
    else:
        plt.plot(arr_accuracy[iAlgo],arr_energy[iAlgo],color=arr_color[iAlgo],label=arr_label[iAlgo])
plt.xlabel("Accuracy",fontsize=16)
#plt.ylabel("energy [arbitrary unit]",fontsize=16)
plt.xlim(0.9,0.98)
plt.ylim(1e3,2e6)
plt.xticks([0.9,0.92,0.94,0.96,0.98])
plt.yticks([1e3,1e4,1e5,1e6])
ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)
plt.yscale("log")
#plt.legend(fontsize=12,loc=2)
plt.tight_layout()
plt.savefig("Plot/energy_vs_accuracy_scatter.png",bbox_inches="tight")
plt.savefig("Plot/energy_vs_accuracy_scatter.svg",bbox_inches="tight")
plt.close()
