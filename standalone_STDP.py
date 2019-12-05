#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import standaloneFunction as sF
import time

tmp = time.gmtime()
np.random.seed(tmp[3]*(tmp[4]*10+tmp[5]))

N_audit = 100 #21
N_visual = 1
N_neurons_each_pool_audit = 15
N_neurons_each_pool_visual = 15
N_b = 1
Cm = 0.5/1e9 # F
El = -70./1e3 # reset/rest potential in V
Ek = -70./1e3 # potassium reversal potential in V
E_fire = -50/1e3 # firing threshold in V
gl = 20./1e9 # second
gb = 1./1e9 # second
dgk = 80./1e9 # delta gK in second
tauk = 110./1e3 # second
taus = 10./1e3 # second
input_rate_b = 1000. # Hz
Eex = 0. # V
gV = 3./1e9 * np.ones(N_visual) # second
gmax = 2.5/1e9 #1.25/1e9 XXX # second
gSTDPsize = 1.25/1e9 # second

tau_plus = 50./1e3 # second
tau_minus = 70./1e3 #110./1e3 XXX # second
A_plus = 0.001
A_minus = 1.05*A_plus*tau_plus/tau_minus

# variables related to how the simulation is run and when to quit
dt = 0.1/1e3
audit_start = 0./1e3 # second
audit_end = 50./1e3 # second
visual_start = 70./1e3 # second
visual_end = 120./1e3 # second
simtime = 0.15 # second
max_run = 1000
buffer_run = 10
no_of_sd = 1 # no_of_sd determines accepted error
BreakLoopV2 = False
BreakLoopV3 = True
accepted_fluctuation = 0.03 * gmax # if the last "buffer_run" events fall within the mean of those events +/- accepted_fluctuation, then the weights have converged

input_rate_audit = 100. # Hz
input_rate_visual = 100. # Hz
input_rate_visual_ratio = [1.0,0.5]
N_active = 2 # number of components in the patterns that are turned on

gA_initial = 0. * np.ones(N_audit)
nPattern = 25

arr_energy = np.zeros(nPattern)
arr_energy_minimal = np.zeros(nPattern)
timesteps = int(simtime/dt)
for iNPattern in range(1,nPattern+1):
    pattern, pattern_answer = sF.MakePattern(iNPattern,N_audit,N_active,input_rate_visual_ratio)
    
    iRun = 0
    gA = np.copy(gA_initial)
    arr_gA = []
    gA_buffer = np.zeros(shape=(N_audit,buffer_run))
    energy_buffer = -999 * np.ones(buffer_run)
    while(iRun < max_run):
        for iPatt in range(iNPattern):
            volt = El
            gk = 0.
            ICX_response = np.zeros(timesteps)
            input_audit = np.zeros(shape=(timesteps, N_audit))
            input_visual = np.zeros(shape=(timesteps, N_visual))
            sA = np.zeros(N_audit); sV = np.zeros(N_visual); sb = 0.
            trace_audit = np.zeros(N_audit)
            trace_ICX = 0.
            gA_previous = np.zeros(len(gA))
            for iTime in range(timesteps):
                if( (iTime*dt > audit_start) and (iTime*dt < audit_end) ):
                    input_audit[iTime] = np.random.binomial(pattern[iPatt]*N_neurons_each_pool_audit, input_rate_audit*dt)
                    sA += input_audit[iTime]
                elif( (iTime*dt > visual_start) and (iTime*dt < visual_end) ):
                    input_visual[iTime] = np.random.binomial(pattern_answer[iPatt]*N_neurons_each_pool_visual, input_rate_visual*dt)
                    sV += input_visual[iTime]
                        
                Il = gl * (volt - El)
                Ik = gk * (volt - Ek)

                sb += np.random.binomial(N_b, input_rate_b*dt)
                Ib = gb * sb * (volt - El)

                Is = ( np.dot(gA,sA) + np.dot(gV,sV) ) * (Eex - volt)
        
                volt += (dt/Cm) * (-Il + Is - Ik + Ib)
                #volt += (dt/Cm) * (-Il + Is - Ik)
                #print(iTime," ",volt*Cm/dt," ",Is," ",Ik," ",gA[0])
                if(volt > E_fire):
                    ICX_response[iTime] = 1
                    volt = El
                    #print("~~~~~~~~ ",iTime*dt," fire")

                # adaptation
                gk += dgk * ICX_response[iTime] - gk * dt / tauk
                if(gk < 0.): gk = 0.


                # do STDP
                trace_audit *= (1-dt/tau_minus)
                trace_ICX *= (1-dt/tau_plus)
                
                tmp = np.where(input_audit[iTime])[0]
                trace_audit[tmp] += np.ones(len(tmp))
                if(ICX_response[iTime]>0):
                    trace_ICX += 1

                gA_previous = np.copy(gA)                    
                if(ICX_response[iTime]>0): # LTP
                    gA += gSTDPsize * A_plus * trace_audit
                gA[tmp] -= gSTDPsize * A_minus * trace_ICX * np.ones(len(tmp)) #LTD
                gA = np.fmax(np.zeros(N_audit),np.fmin(gmax*np.ones(N_audit),gA))

                
                #if(len(np.where(gA>0)[0])>0): print(iTime*dt," ",gA)
                sA -= sA * dt / taus
                sV -= sV * dt / taus
                sb -= sb * dt / taus

                arr_energy[iNPattern-1] += np.sum(np.abs(gA-gA_previous))
            #print(np.mean(input_audit),' ',np.mean(ICX_response[:int(audit_end/dt)]),' ',np.mean(ICX_response[int(visual_start/dt):]))
            #print(np.dot(gA,pattern[iPatt]),' ',iPatt,' ',pattern_answer[iPatt])

        arr_gA.append(np.copy(gA))
        if(BreakLoopV2):
            gA_buffer, bool_converged = sF.BreakLoopV2(gA, gA_buffer, iRun)
        elif(BreakLoopV3):
            gA_buffer, bool_converged = sF.BreakLoopV3(gA, gA_buffer, iRun, accepted_fluctuation=accepted_fluctuation)
        else:
            gA_buffer, bool_converged = sF.BreakLoop(gA, gA_buffer, iRun, no_of_sd=no_of_sd)
        if(bool_converged):
            if(BreakLoopV2):
                arr_energy[iNPattern-1] = arr_energy[iNPattern-1]
                arr_energy_minimal[iNPattern-1] = np.sum( np.fabs( gA-gA_initial ) )
            else:
                arr_energy[iNPattern-1] = energy_buffer[(iRun+1)%buffer_run]
                arr_energy_minimal[iNPattern-1] = np.sum( np.fabs( arr_gA[-buffer_run]-gA_initial ) )
            break
        energy_buffer[iRun%buffer_run] = arr_energy[iNPattern-1]
        iRun += 1
        if((iRun%20)==0): print("patt:", iNPattern,"  run: ",iRun)
    if(not bool_converged):
        arr_energy_minimal[iNPattern-1] = np.sum( np.fabs( arr_gA[-1]-gA_initial ) )
    np.savetxt("Text/20191202/gA_"+str(iNPattern)+".txt",arr_gA)
    np.savetxt("Text/20191202/pattern_"+str(iNPattern)+".txt",pattern)
    np.savetxt("Text/20191202/pattern_answer_"+str(iNPattern)+".txt",pattern_answer)

np.savetxt("Text/20191202/energy.txt",arr_energy)
np.savetxt("Text/20191202/energy_minimal.txt",arr_energy_minimal)
np.savetxt("Text/20191202/other_variables.txt",(N_audit,nPattern))
