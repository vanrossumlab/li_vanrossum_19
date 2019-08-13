# li_vanrossum_19
This contains the simulation code to our paper.

List of files perceptron code to produce the data for Figure 1.
- cperc.m: Octave file for simulating perceptron learning
- runc.cc: For efficiency, the actual learning is done in this C code.
- Makefile: compiles runc.cc

The Octave script runs multi-threaded using pararrayfun;
replacing it with arrayfun.
Conversion to Matlab should be straightforward and is left as an exercise.
