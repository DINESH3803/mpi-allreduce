cat << 'EOF' > run_rdma.sh
#!/bin/sh
#PBS -N rdma_bench
#PBS -l nodes=2:ppn=1
#PBS -l walltime=00:05:00
#PBS -j oe
TARGET_DIR="<directory>"
cd $TARGET_DIR
echo "Running on compute nodes:"
cat $PBS_NODEFILE
echo "Working directory: $(pwd)"
echo "--------------------------------"
module load intel-2018
mpirun -genv I_MPI_DEBUG=5 -genv I_MPI_FABRICS=ofa -np 2 -ppn 1 -f $PBS_NODEFILE $TARGET_DIR/mpi-latency-rdma 0 1
EOF
