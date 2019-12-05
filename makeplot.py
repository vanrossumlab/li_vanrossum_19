#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axisartist.axislines import Subplot

nFile = 30

separation_sd = 5
within_sd = 2

N_audit, nPattern = np.loadtxt("Text/20191202_4/other_variables.txt")

arr_energy = np.zeros(shape=(nFile,int(nPattern)))
arr_energy_minimal = np.zeros(shape=(nFile,int(nPattern)))
for iFile in range(nFile):
    arr_energy[iFile] = np.loadtxt("Text/20191202_"+str(iFile)+"/energy.txt")[:int(nPattern)]
    arr_energy_minimal[iFile] = np.loadtxt("Text/20191202_"+str(iFile)+"/energy_minimal.txt")[:int(nPattern)]


arr_energy_pass = []
arr_energy_minimal_pass = []
for iNPattern in range(1,int(nPattern+1)):
    arr_energy_pass.append([])
    arr_energy_minimal_pass.append([])

arr_audit_response = []
for iFile in range(nFile):
    arr_audit_response.append([])
    for iNPattern in range(1,int(nPattern+1)):
        if(iNPattern==1):
            pattern = [np.loadtxt("Text/20191202_"+str(iFile)+"/pattern_"+str(iNPattern)+".txt")]
            pattern_answer = [np.loadtxt("Text/20191202_"+str(iFile)+"/pattern_answer_"+str(iNPattern)+".txt")]
        else:
            pattern = np.loadtxt("Text/20191202_"+str(iFile)+"/pattern_"+str(iNPattern)+".txt")
            pattern_answer = np.loadtxt("Text/20191202_"+str(iFile)+"/pattern_answer_"+str(iNPattern)+".txt")
        gA = np.loadtxt("Text/20191202_"+str(iFile)+"/gA_"+str(iNPattern)+".txt")

        arr_audit_response[iFile].append([])
        for iPatt in range(iNPattern):
            arr_audit_response[iFile][iNPattern-1].append(np.dot(gA[-1],pattern[iPatt]))


        # check if the network can distinguish between TWO different visual input
        # all auditory response has to be at least (separation_sd*SD) away from the mean of the response from a different visual input
        # all auditory response from the same visual input has to be within (within_sd*SD) from the mean of the response from the same visual input
        nTarget = np.unique(pattern_answer)
        if(iNPattern==1):
            arr_energy_pass[iNPattern-1].append(arr_energy[iFile][iNPattern-1])
            arr_energy_minimal_pass[iNPattern-1].append(arr_energy_minimal[iFile][iNPattern-1])
        elif(len(nTarget)==2):
            tmp = np.where(pattern_answer==nTarget[0])[0]
            if(len(tmp)>0):
                group0 = np.array(arr_audit_response[iFile][iNPattern-1])[tmp]
            else:
                group0 = np.nan
            tmp = np.where(pattern_answer==nTarget[1])[0]
            if(len(tmp)>0):
                group1 = np.array(arr_audit_response[iFile][iNPattern-1])[tmp]
            else:
                group1 = np.nan
            mean0 = np.mean(group0)
            mean1 = np.mean(group1)
            sd0 = np.sqrt(np.var(group0))
            sd1 = np.sqrt(np.var(group1))
            #print iNPattern,' ',sd0,' ',sd1
            bool_pass = True
            if((group0 is not np.nan) and (group1 is not np.nan)):
                for iPatt in range(len(group0)):
                    if( (group0[iPatt] < (mean1+separation_sd*sd1)) and (group0[iPatt] > (mean1-separation_sd*sd1)) ):
                        bool_pass = False
                    if( (group0[iPatt] > (mean0+within_sd*sd0)) or (group0[iPatt] < (mean0-within_sd*sd0)) ):
                        bool_pass = False
                for iPatt in range(len(group1)):
                    if( (group1[iPatt] < (mean0+separation_sd*sd0)) and (group1[iPatt] > (mean0-separation_sd*sd0)) ):
                        bool_pass = False
                    if( (group1[iPatt] > (mean1+within_sd*sd1)) or (group1[iPatt] < (mean1-within_sd*sd1)) ):
                        bool_pass = False
            if(bool_pass):
                arr_energy_pass[iNPattern-1].append(arr_energy[iFile][iNPattern-1])
                arr_energy_minimal_pass[iNPattern-1].append(arr_energy_minimal[iFile][iNPattern-1])
    

mu = np.array([np.nan]*int(nPattern))
up_error = np.array([np.nan]*int(nPattern))
low_error = np.array([np.nan]*int(nPattern))
mu_minimal = np.array([np.nan]*int(nPattern))
up_error_minimal = np.array([np.nan]*int(nPattern))
low_error_minimal = np.array([np.nan]*int(nPattern))
fig = plt.figure(facecolor="white")
ax = Subplot(fig,111)
fig.add_subplot(ax)
for iPatt in range(int(nPattern)):
    energy = np.array(arr_energy_pass[iPatt])
    energy_minimal = np.array(arr_energy_minimal_pass[iPatt])
    if(len(energy)>=2):
        mu[iPatt] = np.mean(energy)
        up_error[iPatt] = np.sqrt( np.sum(np.fabs(energy[np.where(energy>mu[iPatt])]-mu[iPatt])**2) / len(np.where(energy>mu[iPatt])[0]) )
        low_error[iPatt] = np.sqrt( np.sum(np.fabs(energy[np.where(energy<mu[iPatt])]-mu[iPatt])**2) / len(np.where(energy<mu[iPatt])[0]) )
        mu_minimal[iPatt] = np.mean(energy_minimal)
        up_error_minimal[iPatt] = np.sqrt( np.sum(np.fabs(energy_minimal[np.where(energy_minimal>mu_minimal[iPatt])]-mu_minimal[iPatt])**2) / len(np.where(energy_minimal>mu_minimal[iPatt])[0]) )
        low_error_minimal[iPatt] = np.sqrt( np.sum(np.fabs(energy_minimal[np.where(energy_minimal<mu_minimal[iPatt])]-mu_minimal[iPatt])**2) / len(np.where(energy_minimal<mu_minimal[iPatt])[0]) )
print(mu)
print(mu_minimal)
plt.errorbar(np.linspace(1,nPattern,nPattern), mu, yerr=[low_error,up_error],
             color="red", linestyle="-", linewidth=3)
plt.errorbar(np.linspace(1,nPattern,nPattern), mu_minimal,
             yerr=[low_error_minimal,up_error_minimal],
             color="green", linestyle="-", linewidth=3)
plt.xlabel("#patterns",fontsize=16)
plt.ylabel("Energy (a.u.)",fontsize=16)
#plt.xlim(0,nPattern)
plt.xlim(0,np.argmax(np.isnan(mu))+1)
plt.ylim(1e-9,1e-6)
plt.yscale("log")
plt.xticks([0,4,8,12,16])
plt.yticks([1e-9,1e-8,1e-7,1e-6])
ax.axis["right"].set_visible(False)
ax.axis["top"].set_visible(False)
ax.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig("Plot/DSouza_energy.png")
plt.savefig("Plot/DSouza_energy.eps")
plt.savefig("Plot/DSouza_energy.svg")
plt.close()
