# Makefile for the shared-memory + OpenMP Allreduce assignment.
#
# Toolchain: Intel MPI 2018 Update 2 (mpicc wraps GCC 4.8.5), OpenMP 3.1.
#   module load intel-2018
#   make                    # build everything
#   make benchmark          # mpi-allreduce-shm-bench only
#   make shared_test        # mpi_shared_test only
#   make clean
#
# If the benchmark reports thread level MPI_THREAD_SINGLE, rebuild with the
# thread-safe Intel MPI library:
#   make clean && make MPIFLAGS=-mt_mpi
#
# sharedlibrary.c is deliberately NOT built or linked: it replaces the
# MPI_Allreduce symbol with a function that does nothing.

CC       = mpicc
MPIFLAGS =
CFLAGS   = -std=gnu99 -O3 -fopenmp -Wall -Wextra $(MPIFLAGS)
LDFLAGS  = -fopenmp $(MPIFLAGS)
LDLIBS   =

BENCH       = mpi-allreduce-shm-bench
SHARED_TEST = mpi_shared_test

.PHONY: all benchmark shared_test clean

all: benchmark shared_test

benchmark: $(BENCH)

shared_test: $(SHARED_TEST)

$(BENCH): mpi-allreduce-shm-bench.o my_allreduce.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

mpi-allreduce-shm-bench.o: mpi-allreduce-shm-bench.c my_allreduce.h
	$(CC) $(CFLAGS) -c -o $@ $<

my_allreduce.o: my_allreduce.c my_allreduce.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(SHARED_TEST): mpi_shared_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(BENCH) $(SHARED_TEST) mpi-allreduce-shm-bench.o my_allreduce.o
