#include <stdio.h>
#include <mpi.h>

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    /* my implementation */
    printf("My own weird allreduce\n");
    return MPI_SUCCESS;
}
