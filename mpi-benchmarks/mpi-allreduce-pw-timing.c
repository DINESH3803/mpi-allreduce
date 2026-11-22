#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#define WARMUP 10
#define ITER 10
#define MAX_SIZE (1024*1024*16)
#define WINDOW 64
int main(int argc, char **argv) {

  int msize,size, rank;
  int *src=NULL, *dst=NULL;
  int *pw_dst=NULL, *tmp=NULL;
  int send_peer, recv_peer;
  int i, j;
  struct timeval start, end;
  float total_time=0;
  int max_size=0;
  MPI_Status status[WINDOW];
  MPI_Request send_request[WINDOW];
  MPI_Request recv_request[WINDOW];


  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  max_size = MAX_SIZE;
  printf("Allreduce benchmark\n");

  /* allocate memory */
  src = (int *)malloc(max_size*sizeof(int));
  dst = (int *)malloc(max_size*sizeof(int));
  pw_dst = (int *)malloc(max_size*sizeof(int));
  tmp = (int *)malloc(max_size*sizeof(int));
  if (src == NULL || dst == NULL || pw_dst == NULL || tmp == NULL) {
     perror("cannot allocate memory\n");
     exit(0);
  }

  for (i=0; i < max_size; i++) {
      src[i] = rank*size + i;
      pw_dst[i] = 0;
  }
     
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce (src, dst, max_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
for(msize=1024*256;msize<MAX_SIZE;msize *=2) {
  for(j=0; j <WARMUP + ITER;j++) {
  if(j== WARMUP) gettimeofday(&start, NULL);
  max_size=msize;
  /* reduce-scatter phase */
  for(i=0;i<size;i++) {
    send_peer = (rank ^ i);
    recv_peer = (rank ^ i);
    MPI_Isend(&src[send_peer*(max_size/size)], max_size/size, MPI_INT, send_peer, 1, MPI_COMM_WORLD, &send_request[i]);
    MPI_Irecv(&tmp[recv_peer*(max_size/size)], max_size/size, MPI_INT, recv_peer, 1, MPI_COMM_WORLD, &recv_request[i]);
  }
  MPI_Waitall(size, send_request, status);
  MPI_Waitall(size, recv_request, status);

  /* reduction */
  #pragma omp parallel for num_threads(2)
  for(i=0;i<size;i++) {
     for(j=0;j<(max_size/size);j++) {
       pw_dst[rank*(max_size/size)+j] += tmp[i*(max_size/size)+j];
     }
  }

  /* allgather phase */
  for(i=0;i<size;i++) {
    send_peer = (rank ^ i);
    recv_peer = (rank ^ i);
    MPI_Isend(&pw_dst[rank*(max_size/size)], max_size/size, MPI_INT, send_peer, 1, MPI_COMM_WORLD, &send_request[i]);
    MPI_Irecv(&pw_dst[recv_peer*(max_size/size)], max_size/size, MPI_INT, recv_peer, 1, MPI_COMM_WORLD, &recv_request[i]);
  }
  MPI_Waitall(size, send_request, status);
  MPI_Waitall(size, recv_request, status);
 }
  gettimeofday(&end, NULL);
  if(!rank) printf("%10d\t%10.4lf\n",msize*4, ((float)(end.tv_sec-start.tv_sec)*1000000.0 + (end.tv_usec-start.tv_usec))/ITER);
  break;
}
  MPI_Finalize();
  return 0;
}

