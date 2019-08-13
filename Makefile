# Makefile 
# exmaple:http://lasp.colorado.edu/cism/CISM_DX/code/CISM_DX-0.50/required_packages/octave-forge/main/sparse/Makefile
MKOCTFILE= mkoctfile

MKOCTFLAGS   = -lgsl -O3 -Wall
SRC = runc.cc 

all: runc.oct runcw.oct 

runc.oct: runc.cc
	$(MKOCTFILE) $(SRC) $(MKOCTFLAGS)
