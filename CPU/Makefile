CC = g++
FLAGS = -g -mpopcnt -fopenmp -O3

FLAGS+=-DTHREADS=$(threads) -DFREQ=$(freq)

ifeq ($(freq),)
test_freq:
	@echo 'Frequency is not defined! Run make freq=<nominal> to set up frequency and measure the time!'
endif


all: v1 v2 v3 v4_avx v4_avx512 v4_popc_avx512

v1: v1.cpp
	$(CC) -o $@ $^ -fno-tree-vectorize $(FLAGS) 

v2: v2.cpp
	$(CC) -o $@ $^ -fno-tree-vectorize $(FLAGS) 

v3: v3.cpp
	$(CC) -o $@ $^ -fno-tree-vectorize $(FLAGS) 

v4_avx: v4_avx.cpp
	$(CC) -o $@ $^ -mavx2 $(FLAGS) 

v4_avx512: v4_avx512.cpp
	$(CC) -o $@ $^ -march=skylake-avx512 $(FLAGS) 

v4_popc_avx512: v4_popc_avx512.cpp
	$(CC) -o $@ $^ -march=icelake-server $(FLAGS) 


clean:
	rm v1 v2 v3 v4_avx v4_avx512 v4_popc_avx512
