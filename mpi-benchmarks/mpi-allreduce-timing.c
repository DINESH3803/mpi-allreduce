#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define WARMUP 10
#define ITER  10
#define MAX_SIZE (1024*1024*16)

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
  printf("Allreduce benchmark\n");

  max_size = MAX_SIZE;
  /* allocate memory */
  src = (int *)malloc(max_size*sizeof(int));
  dst = (int *)malloc(max_size*sizeof(int));
  if (src == NULL || dst == NULL) {
     perror("cannot allocate memory\n");
     exit(0);
  }

  for (i=0; i < max_size; i++) {
      src[i] = rank;
  }
     
  MPI_Barrier(MPI_COMM_WORLD);
  for(size=1024*256;size<MAX_SIZE;size *=2) {
    for(i=0; i <WARMUP + ITER;i++) {
       if(i== WARMUP) gettimeofday(&start, NULL);
       MPI_Allreduce (src, dst, size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    gettimeofday(&end, NULL);
    if(!rank) printf("%10d\t%10.4lf\n",size*4, ((float)(end.tv_sec-start.tv_sec)*1000000.0 + (end.tv_usec-start.tv_usec))/ITER);
  }  
  
  MPI_Finalize();
  return 0;
}

