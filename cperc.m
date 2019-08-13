% plasticity energy of perceptron

clear all
close all
pkg load parallel

function [pmat,targets]=make_pats(N,P,ppat=0.5,qpat=0.5, zeromeanQ=1)
    % random patterns
    pmat=2*(rand(N+1,P)<ppat)-1;
    
   if (zeromeanQ==1)
        pmat -= mean(mean(pmat));
    end    
    pmat(N+1,:)=1; % for bias input

    % random binary +- 1 target outputs
    targets = -1+2*(rand(1,P)<qpat);
end


function  [epochs,succes,Ectrl,Emin,El,Ee]=runc_wrapperP(P,N,i_algol,tau,maxepochs,ethr,ppat=0.5,qpat=0.5,zeromeanQ=1)
% every call uses different patterns
    [pmat,targets]=make_pats(N,P,ppat,qpat,zeromeanQ);

    [epochs,succes,Ectrl,Emin,El,Ee]= runc(pmat,targets, i_algol, tau, maxepochs, ethr);
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ret=system('make');
if (ret!=0)
    error('Stopping: compilation error')
end

res=[];
maxepochs = 10000; % max epochs before deciding non-convergence

    for N = 1000 % 200

        P_ar=10:1:1999;
        i_algol=1;
        tau=0;
        ethr=0;

        [epochs,succes,Ectrl,Emin,El,Ee]= ...
            pararrayfun(nproc,@(P)         ...
                runc_wrapperP(P,N, i_algol, tau, maxepochs,ethr), P_ar);

        % filter out non-converging
        epochs=epochs(find(succes==1));
        Emin=Emin(find(succes==1));
        El=El(find(succes==1));
        P_ar=P_ar(find(succes==1))/N;
        
        nr_nonconverging=sum(succes==0)
      
        semilogy(P_ar,El)
        hold on
        semilogy(P_ar,epochs)

        tmp=[P_ar;El]';
        save El.dat tmp
         
        tmp=[P_ar;Emin]';
        save Emin.dat tmp
         
        tmp=[P_ar;epochs]';
        save epochs.dat tmp
            
    end % N 
    
