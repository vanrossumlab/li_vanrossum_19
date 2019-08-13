#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import Subplot

nFile = 3
no_of_hidden_nodes = np.loadtxt('Text/nHidden_new_0/nHidden_no_of_hidden_nodes.txt')
learning_rate,save_frequency,decay_eLTP,energy_scale_maintenance = np.loadtxt('Text/nHidden_new_0/nHidden_other_variable.txt')
accuracy_required = np.loadtxt('Text/nHidden_new_0/nHidden_accuracy_required.txt')

standard_energy = np.nan*np.ones(shape=(len(accuracy_required),len(no_of_hidden_nodes),nFile))
caching_energy = np.nan*np.ones(shape=(len(accuracy_required),len(no_of_hidden_nodes),nFile))
for iFile in range(nFile):
    for iAccu in range(len(accuracy_required)):
        if((iAccu==1)and(iFile==2)):
            continue
        nHidden_performed = np.loadtxt('Text/nHidden_new_'+str(iFile)+'/nHidden_nHidden_AlgoStandard_accuracy'+str(accuracy_required[iAccu])+'.txt')
        tmp_standard_energy = np.loadtxt('Text/nHidden_new_'+str(iFile)+'/nHidden_energy_AlgoStandard_accuracy'+str(accuracy_required[iAccu])+'.txt')
        for index in range(len(nHidden_performed)):
            iNode = np.where(no_of_hidden_nodes==nHidden_performed[index])[0][0]
            standard_energy[iAccu][iNode][iFile] = tmp_standard_energy[iNode]
            tmp_accuracy = np.loadtxt('Text/nHidden_new_'+str(iFile)+'/nHidden_accuracy_AlgoCaching_accuracy'+str(accuracy_required[iAccu])+'_nHidden'+str(int(nHidden_performed[index]))+'.txt')
            tmp_energy = np.loadtxt('Text/nHidden_new_'+str(iFile)+'/nHidden_energy_AlgoCaching_accuracy'+str(accuracy_required[iAccu])+'_nHidden'+str(int(nHidden_performed[index]))+'.txt')
            tmp = np.where( tmp_accuracy > accuracy_required[iAccu] )[0]
            if(len(tmp) > 0):
                caching_energy[iAccu][iNode][iFile] = np.min(tmp_energy[tmp])
            
standard_average_energy = np.nan*np.ones(shape=(len(accuracy_required),len(no_of_hidden_nodes)))
caching_average_energy = np.nan*np.ones(shape=(len(accuracy_required),len(no_of_hidden_nodes)))
for iAccu in range(len(accuracy_required)):
    for iNode in range(len(no_of_hidden_nodes)):
        tmp = ~np.isnan(standard_energy[iAccu][iNode])
        standard_average_energy[iAccu][iNode] = np.mean(standard_energy[iAccu][iNode][tmp])
        tmp = ~np.isnan(caching_energy[iAccu][iNode])
        caching_average_energy[iAccu][iNode] = np.mean(caching_energy[iAccu][iNode][tmp])


arr_color = ['red','black']
arr_linestyle = ['-','--',':']
arr_label = ['No caching','Synaptic caching']

fig = plt.figure(facecolor="white",figsize=(4.5,4))
ax = Subplot(fig,111)
fig.add_subplot(ax)
for iAccu in range(len(accuracy_required)):
    plt.plot(no_of_hidden_nodes,standard_average_energy[iAccu],color=arr_color[0],linewidth=3,linestyle=arr_linestyle[iAccu],label=arr_label[0]+', '+str(accuracy_required[iAccu]))
    plt.plot(no_of_hidden_nodes,caching_average_energy[iAccu],color=arr_color[1],linewidth=3,linestyle=arr_linestyle[iAccu],label=arr_label[1]+', '+str(accuracy_required[iAccu]))
plt.xlabel('# hidden units',fontsize=24)
#plt.ylabel('Energy [arbitrary unit]',fontsize=28)
plt.yscale('log')
plt.xlim(0,no_of_hidden_nodes[0])
#plt.xticks(np.linspace(0,10,11))
plt.ylim(1e3,1e6)
plt.xticks([0,50,100,150,200])
plt.yticks([1e3,1e4,1e5,1e6])
ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)
#ax.axis["bottom"].label.set_text('# hidden units')
ax.tick_params(labelsize=20)
#plt.legend(bbox_to_anchor=(0.,1.02,1.,0.112),fontsize=10,loc='lower left',ncol=2,mode='expand',borderaxespad=0.)
plt.tight_layout()
plt.savefig('Plot/nHidden.png',bbox_inches='tight')
plt.savefig('Plot/nHidden.svg',bbox_inches='tight')
plt.close()
