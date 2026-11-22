#!/bin/sh
#PBS -N mpi-latency_original
#PBS -l nodes=2:ppn=1
#PBS -l walltime=00:08:00
#PBS -j oe

module load intel-2018

cd /home/student4/mpi-benchmarks

mpirun -genv I_MPI_DEBUG=5 -genv I_MPI_FABRICS=ofa -np 2 -ppn 1 -f $PBS_NODEFILE ./mpi-latency_original 0 1
