#!/bin/bash
#PBS -N nhpc_bench
#PBS -q short
#PBS -l nodes=2:ppn=1
#PBS -V
cd /home/student4/mpi-benchmarks
source /opt/apps/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/bin/mpivars.sh
/opt/apps/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/bin/mpirun -genv I_MPI_FABRICS=ofa ./mpi-latency
