# li_vanrossum_19
This contains the simulation code to our paper 
  Energy efficient synaptic plasticity. eLife 2020 Feb 13;9:e50804. doi: 10.7554/eLife.50804.

List of files perceptron code to produce the data for Figure 1 and can be edited to produce the data for the other figures related to the perceptron.
- cperc.m: Octave file for simulation of perceptron learning
- runc.cc: For efficiency, the actual learning is done in this C code.
- Makefile: compiles runc.cc

The Octave script runs multi-threaded using pararrayfun; this might have broken in recent version of pararrayfun in Octave
replacing it with arrayfun will run single threaded.
Conversion to Matlab should be straightforward and is left as an exercise.

List of multi-layer codes to produce Figure 4. Each panel in the figure has a specific code to generate its data, which then is plotted by a respective "MakePlot" Python code.
- learning_rate.py: generating data for panel (a), and its plotting code is learning_rate_MakePlot.py
- energy_vs_accuracy.py: for (b), the plotting code is energy_vs_accuracy_MakePlot.py
- nHidden.py: for (c), the plotting code is nHidden_MakePlot.py
- KleinFunction.py: this contains the function that conducts the back-propagation training, and is called by the Python codes above.

List of codes to produce the energy cost of learning by using D'Souza et al. (2010) learning rule (Figure 2-Figure 1 Supplement):
- standalone_STDP.py: generating data
- standaloneFunction.py: contains functions that are called by standalone_STDP.py
- makeplot.py: for analysing the data and plotting the result.
