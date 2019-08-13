// to be called from Octave
// compile with make, or mkoctfile runc.cc -lgsl -O3

#include <octave/oct.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <time.h>

// Arguments: pattern matrix, learning rate, algorithm, decay, maxiter, eLTP-threshold.

DEFUN_DLD(runc,args,nargout,"help? no way"){
    int nargin = args.length (); 
    
    // octave_stdout << "Receive " << nargin << " input args, " << nargout << " output arguments.\n";

    int nargs_required=6;                
    if (nargin != nargs_required){
        octave_stdout << "Incorrect number of arguments \n ";
        return octave_value_list ();
    }
    
    NDArray pmat    = args(0).array_value(); // dim N+1 x P
    NDArray targets = args(1).array_value(); // dim  P
    int i_agol      = args(2).int_value();
    
    double tau_decay= args(3).double_value();
    int maxepochs   = args(4).int_value();
    double ethr     = args(5).double_value();
    
    double decayfact= 1-1/tau_decay;

    dim_vector dv=pmat.dims();       
    int N = dv(0)-1; // account for bias term
    int P = (int)dv(1);
    
    int succes=0;
       
    double El   =0.;
    double Ee   =0.;
    double Ectrl=0.; // default algorithm
   
    // Using NDarrays has more functions: /usr/include/octave-4.4.1/octave/dNDArray.h
    dim_vector dv2 (N+1,1);
    NDArray we(dv2);
    NDArray wl(dv2);
    NDArray dw(dv2);
    
    RowVector updates(maxepochs);
    
    we.fill(0.);
    wl.fill(0.);
    
//   octave_stdout << "N= " << N <<", P= " << P <<"\n";
 //   octave_stdout << "decay= " << decayfact <<"\n";
   
    int iepoch;
    int timecounter =1;
    int updatecounter =1;
    for (iepoch=0; iepoch < maxepochs; iepoch++){
        //octave_stdout << iepoch << "\n";
        int errors=0;
        for (int ipat=0;  ipat < P; ipat++){
            timecounter ++;
            if (tau_decay !=0.0){
                we *= decayfact;
            }
            
            // get output and update Ee
            double y=0.0;
            for (int i = 0; i<N+1; i++){
                y+= (wl(i)+we(i))*pmat(i,ipat);
                Ee += abs(we(i));
            }
            
            // check and learn 
            if (y*targets(ipat)<=0) {
                errors++;
                for (int i =0 ; i<N+1; i++){
                    dw(i)= targets(ipat)*pmat(i,ipat); 
                    //  dw(i)= eps*targets(ipat)*pmat(i,ipat); 
                    Ectrl += abs(dw(i)); // when tau!=0, this is not equal to E_lLTP-only
                }
                
                if (i_agol==1){ // no early phase, classic learning
                    wl += dw;
                    updatecounter++;
                    for (int i=0 ; i<N+1; i++){
                        El += abs(dw(i));
                    }
                }else if (i_agol==2){ // indiv. thr, indiv consolod
                    we += dw;
                    for (int i=0 ; i <N+1; i++){
                        if (abs(we(i)) > ethr) {    // consolidate
                            wl(i)   += we(i);
                            El      += abs(we(i));
                            we(i)   =  0.0;
                        }
                    }
                }else if (i_agol==3){ // indiv. thr, glob consol.
                    we += dw;
                    double m = (we.abs()).max()(0);
                    if (m > ethr){
                        for (int i=0 ; i <N+1; i++){
                            wl(i)   += we(i);
                            El      += abs(we(i));
                            we(i)   =  0.0;
                        }
                    } 
                }else if (i_agol==4){ // global thr, glob consol.
                    we += dw;
                    double m = (we.abs()).sum()(0); // normalize? (1/N or sqrtN ?)
                    if (m > ethr){
                        for (int i=0 ; i <N+1; i++){
                            wl(i)   += we(i);
                            El      += abs(we(i));
                            we(i)   =  0.0;
                        }
                    } 
                }else if (i_agol==5){ // L1 norm
                    for (int i=0 ; i <N+1; i++){
                        double dw2;
                        if (wl(i)>0){
                            dw2 = dw(i)-ethr;
                        }else{
                            dw2 = dw(i)+ethr;
                        }
                        wl(i) += dw2;
                        El += abs(dw2);
                    }
                }
                              
                else{
                    octave_stdout << "No such alogirthm \n";
                    break;
                }
                    
            } // end if update   
                        
        } // ipat
        
        
        //octave_stdout << "errors"<< errors<< "\n";
        updates(iepoch)= errors;
        
        if (errors ==0){
            succes=1;
            break;
        }
    }//iepoch
    
    if (succes==0){
        octave_stdout << "no convergence, ethr = " << ethr << "\n";
    }
    
    // final consolidation:
    double Emin=0.;
    bool skip_energy_finalQ=false;
    for (int i=0; i< N+1; i++) {
        wl(i) += we(i);
        if (!skip_energy_finalQ){
            El    += abs(we(i));
        }
        Emin  += abs(wl(i));
        we(i) =  0.;
    }
        
    octave_value_list retval;
    int i=0;
    retval(i++) = octave_value (iepoch);
    retval(i++) = octave_value (succes);
    retval(i++) = octave_value (Ectrl);
    retval(i++) = octave_value (Emin);
    retval(i++) = octave_value (El);
    retval(i++) = octave_value (Ee);
   
    return retval;
}

