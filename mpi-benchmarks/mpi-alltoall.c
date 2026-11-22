#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char **argv) {

  int size, rank;
  int *src=NULL, *dst=NULL;
  int i;
  MPI_Status status;
  struct timeval start, end;
  float total_time=0;
  int max_size=0;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  max_size = size;
  printf("All2All benchmark\n");

  /* allocate memory */
  src = (int *)malloc(max_size*sizeof(int));
  dst = (int *)malloc(max_size*sizeof(int));
  if (src == NULL || dst == NULL) {
     perror("cannot allocate memory\n");
     exit(0);
  }

  for (i=0; i < max_size; i++) {
      src[i] = rank;
      printf("rank=%d src index=%d has value %2d\n", rank, i, src[i]);
  }
     
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Alltoall (src, max_size/size, MPI_INT, dst, max_size/size, MPI_INT, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  for (i=0; i < max_size; i++) {
      printf("rank=%d dst index=%d has value %2d\n", rank, i, dst[i]);
  }
  
  MPI_Finalize();
  return 0;
}

