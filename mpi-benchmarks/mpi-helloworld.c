#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {

  int size, rank;
  char hostname[256];
  MPI_Init(&argc, &argv);

  /* Total number of MPI processes in the job */
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Get the rank ID of this MPI process */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  gethostname(&hostname,256);

  printf("[%s]: Hello world size=%d rank=%d\n", hostname, size, rank);

  MPI_Finalize();
  return 0;
}

