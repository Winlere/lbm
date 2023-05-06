# Makefile
CC = gcc
CFLAGS = -std=c99 -Wall -O3
LIBS = -lm -fopenmp
SRC = main.c d2q9_bgk.c calc.c utils.c 
EXE=lbm

PARAMS_DIR=./data/params
OBSTACLES_DIR=./data/obstacles
RESULTS_DIR=./results
SCRIPTS_DIR=./scripts

ifeq ($(LBM_PERF), 1)
	CFLAGS += -pg -g
endif

.PHONY: all visual check clean

all: clean
	$(CC) $(CFLAGS) $(SRC) $(LIBS) -o $(EXE) 

visual: clean
	rm -rf $(RESULTS_DIR)/visual
	mkdir -p $(RESULTS_DIR)/visual
	$(CC) $(CFLAGS) $(SRC) $(LIBS) -o $(EXE) -DVISUAL
	./$(EXE) $(PARAMS_DIR)/visual.params $(OBSTACLES_DIR)/CS110.dat
	gnuplot $(SCRIPTS_DIR)/visual.plt

evaluate: all
	mkdir -p $(RESULTS_DIR)
	./$(EXE) $(PARAMS_DIR)/evaluate.params $(OBSTACLES_DIR)/CS110.dat 

plot:
	gnuplot $(SCRIPTS_DIR)/final_state.plt

clean:
	rm -f $(EXE)
	rm -f *.dat