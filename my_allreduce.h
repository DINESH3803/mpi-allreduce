/*
 * my_allreduce.h
 *
 * Single-node Allreduce (MPI_INT, MPI_SUM) built on an MPI-3 shared-memory
 * window, with OpenMP threads doing the memory copies and the reduction.
 *
 * Usage (all calls are collective over the communicator given to init):
 *
 *     MPI_Init_thread(..., MPI_THREAD_FUNNELED, &provided);
 *     my_allreduce_init(MPI_COMM_WORLD, max_count);
 *     my_mpi_allreduce(sendbuf, recvbuf, count);     (count <= max_count)
 *     ...
 *     my_allreduce_finalize();
 *     MPI_Finalize();
 *
 * Rules for callers:
 *   - MPI must already be initialised with at least MPI_THREAD_FUNNELED.
 *   - Call these functions from the MPI main thread, outside any OpenMP
 *     parallel region. The library opens its own parallel regions; their
 *     size follows OMP_NUM_THREADS / omp_set_num_threads().
 *   - Every rank of the communicator must make the same sequence of calls
 *     with the same count, exactly like MPI collectives.
 *   - All ranks of the communicator must be on one node.
 *   - Only one instance can be active at a time (global state, not thread-safe).
 */
#ifndef MY_ALLREDUCE_H
#define MY_ALLREDUCE_H

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Return codes.
 *   0         success (equal to MPI_SUCCESS)
 *   > 0       an MPI error code, passed through unchanged from a failed MPI call
 *   < 0       a library error, listed below
 */
#define MY_ALLREDUCE_SUCCESS               0
#define MY_ALLREDUCE_ERR_NOT_INITIALIZED (-1)  /* my_allreduce_init not called    */
#define MY_ALLREDUCE_ERR_ALREADY_INIT    (-2)  /* my_allreduce_init called twice  */
#define MY_ALLREDUCE_ERR_MPI_NOT_INIT    (-3)  /* MPI_Init_thread not called yet  */
#define MY_ALLREDUCE_ERR_THREAD_LEVEL    (-4)  /* provided < MPI_THREAD_FUNNELED  */
#define MY_ALLREDUCE_ERR_NOT_MAIN_THREAD (-5)  /* called from a non-main thread   */
#define MY_ALLREDUCE_ERR_IN_PARALLEL     (-6)  /* called inside an OpenMP region  */
#define MY_ALLREDUCE_ERR_BAD_ARG         (-7)  /* invalid count / NULL buffer     */
#define MY_ALLREDUCE_ERR_ARG_MISMATCH    (-8)  /* ranks passed different counts   */
#define MY_ALLREDUCE_ERR_NOT_SINGLE_NODE (-9)  /* communicator spans >1 node      */
#define MY_ALLREDUCE_ERR_NO_MEMORY      (-10)  /* malloc failed / size too large  */
#define MY_ALLREDUCE_ERR_REMOTE         (-11)  /* another rank reported an error  */
#define MY_ALLREDUCE_ERR_BAD_SEGMENT    (-12)  /* unexpected shared-segment query */

/*
 * Collective. Builds the node-local communicator and allocates the shared
 * window, sized for messages of up to max_count ints.
 * Every rank must pass the same max_count (> 0).
 */
int my_allreduce_init(MPI_Comm comm, int max_count);

/*
 * Collective. recvbuf[i] = sum over all ranks of sendbuf[i], for
 * 0 <= i < count, with 0 <= count <= max_count. Every rank receives the
 * complete result. sendbuf may equal recvbuf. MPI_IN_PLACE is not supported.
 * The sum is plain int addition: the caller must make sure it cannot overflow.
 */
int my_mpi_allreduce(const int *sendbuf, int *recvbuf, int count);

/*
 * Collective. Frees the window and the communicator. Call before MPI_Finalize.
 */
int my_allreduce_finalize(void);

/*
 * Returns a readable message for a return code from the functions above.
 * Uses a static buffer for MPI error codes, so it is not thread-safe.
 */
const char *my_allreduce_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif /* MY_ALLREDUCE_H */
