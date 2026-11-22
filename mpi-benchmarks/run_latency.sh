#!/bin/sh
#PBS -l nodes=2:ppn=4
#PBS -l walltime=00:08:00

cd /home/student4/mpi-benchmarks

echo "Allocated nodes:"
cat $PBS_NODEFILE

echo "Testing MPI rank placement:"
mpirun -np 8 -ppn 4 -f $PBS_NODEFILE hostname

echo "Running benchmark:"
module load intel-2018

mpirun -genv I_MPI_DEBUG=5 \
       -genv I_MPI_FABRICS=ofa \
       -np 8 \
       -ppn 4 \
       -f $PBS_NODEFILE \
       ./mpi-latency
