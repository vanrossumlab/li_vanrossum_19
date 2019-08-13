# li_vanrossum_19
This contains the simulation code to our paper.

List of files perceptron code to produce the data for Figure 1 and can be edited to produce the data for the other figures related to the perceptron.
- cperc.m: Octave file for simulating perceptron learning
- runc.cc: For efficiency, the actual learning is done in this C code.
- Makefile: compiles runc.cc

The Octave script runs multi-threaded using pararrayfun;
replacing it with arrayfun.
Conversion to Matlab should be straightforward and is left as an exercise.

List of multi-layer codes to produce Figure 4. Each panel in the figure has a specific code to generate its data, which then is plotted by a respective "MakePlot" Python code.
- learning_rate.py: generating data for panel (a), and its plotting code is learning_rate_MakePlot.py
- energy_vs_accuracy.py: for (b), the plotting code is energy_vs_accuracy_MakePlot.py
- nHidden.py: for (c), the plotting code is nHidden_MakePlot.py
- KleinFunction.py: this contains the function that conducts the back-propagation training, and is called by the Python codes above.
