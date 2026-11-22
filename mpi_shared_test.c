/*
 * mpi_shared_test.c
 *
 * Temporary sanity test for MPI-3 shared-memory windows on a single node.
 *
 *   - MPI_Init_thread(MPI_THREAD_FUNNELED) and report the provided level
 *   - MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) -> node-local communicator
 *   - MPI_Win_allocate_shared: one int per rank
 *   - MPI_Win_shared_query: direct load/store pointers to every rank's int
 *   - each rank stores its (node-local) rank into its own slot
 *   - MPI_Win_lock_all + MPI_Win_sync + MPI_Barrier for memory consistency
 *   - each rank reads and prints every slot and verifies it
 *
 * Written for GCC 4.8.5 (default -std=gnu90) and Intel MPI 2018 Update 2:
 * declarations at block start, no for-loop declarations, C-style comments.
 *
 * Build:  mpicc -O2 -Wall -o mpi_shared_test mpi_shared_test.c
 * Run:    mpirun -genv I_MPI_FABRICS=shm -np 4 ./mpi_shared_test
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int g_world_rank = -1;

/* Report an MPI error with its text and abort the whole job. */
static void mpi_fail(int rc, const char *call, const char *file, int line)
{
    char msg[MPI_MAX_ERROR_STRING];
    int  len = 0;

    if (MPI_Error_string(rc, msg, &len) != MPI_SUCCESS) {
        sprintf(msg, "unknown error code %d", rc);
    }
    fprintf(stderr, "[world rank %d] MPI error in %s at %s:%d: %s\n",
            g_world_rank, call, file, line, msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, (rc != 0) ? rc : 1);
}

#define MPI_CHECK(call)                                              \
    do {                                                             \
        int rc_ = (call);                                            \
        if (rc_ != MPI_SUCCESS) {                                    \
            mpi_fail(rc_, #call, __FILE__, __LINE__);                \
        }                                                            \
    } while (0)

static const char *thread_level_name(int level)
{
    switch (level) {
    case MPI_THREAD_SINGLE:     return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED:   return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE:   return "MPI_THREAD_MULTIPLE";
    default:                    return "UNKNOWN";
    }
}

int main(int argc, char **argv)
{
    int provided = -1;
    int rc;
    int world_size = 0;
    int shm_rank = -1, shm_size = 0;
    MPI_Comm shm_comm = MPI_COMM_NULL;
    MPI_Win  win = MPI_WIN_NULL;
    int *my_slot = NULL;
    volatile int **slots = NULL;   /* volatile: force real loads from shared memory */
    int *values = NULL;
    int  local_errors = 0, global_errors = 0;
    int  r, turn;

    /* ---- 1. Initialise MPI with FUNNELED thread support ---------------- */
    rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed with code %d\n", rc);
        return EXIT_FAILURE;
    }

    /* Return error codes instead of aborting inside the library, so that
     * MPI_CHECK can print which call failed. Communicators created from
     * MPI_COMM_WORLD below inherit this handler. */
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_world_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    if (g_world_rank == 0) {
        printf("Requested thread level: %s\n", thread_level_name(MPI_THREAD_FUNNELED));
        printf("Provided  thread level: %s (%d)\n", thread_level_name(provided), provided);
        if (provided < MPI_THREAD_FUNNELED) {
            printf("WARNING: MPI_THREAD_FUNNELED not available; "
                   "OpenMP regions must not overlap MPI calls.\n");
        }
        fflush(stdout);
    }

    /* ---- 2. Node-local shared-memory communicator ---------------------- */
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  0 /* key: keep world order */,
                                  MPI_INFO_NULL, &shm_comm));
    MPI_CHECK(MPI_Comm_rank(shm_comm, &shm_rank));
    MPI_CHECK(MPI_Comm_size(shm_comm, &shm_size));

    if (g_world_rank == 0) {
        printf("World size: %d, shared-memory communicator size: %d\n",
               world_size, shm_size);
        if (shm_size != world_size) {
            printf("NOTE: ranks span more than one node; "
                   "this test checks the node of each rank separately.\n");
        }
        fflush(stdout);
    }

    /* ---- 3. Allocate one int per rank in a shared window --------------- */
    MPI_CHECK(MPI_Win_allocate_shared((MPI_Aint)sizeof(int), (int)sizeof(int),
                                      MPI_INFO_NULL, shm_comm,
                                      &my_slot, &win));
    MPI_CHECK(MPI_Win_set_errhandler(win, MPI_ERRORS_RETURN));

    slots  = (volatile int **)malloc((size_t)shm_size * sizeof(*slots));
    values = (int *)malloc((size_t)shm_size * sizeof(int));
    if (slots == NULL || values == NULL) {
        fprintf(stderr, "[world rank %d] cannot allocate %d pointers\n",
                g_world_rank, shm_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- 4. Get direct pointers to every rank's slot ------------------- */
    for (r = 0; r < shm_size; r++) {
        MPI_Aint seg_size = 0;
        int      disp_unit = 0;
        int     *ptr = NULL;

        MPI_CHECK(MPI_Win_shared_query(win, r, &seg_size, &disp_unit, &ptr));
        if (ptr == NULL || seg_size < (MPI_Aint)sizeof(int) ||
            disp_unit != (int)sizeof(int)) {
            fprintf(stderr, "[world rank %d] bad shared segment for rank %d: "
                    "ptr=%p size=%ld disp_unit=%d\n",
                    g_world_rank, r, (void *)ptr, (long)seg_size, disp_unit);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        slots[r] = ptr;
    }

    /* ---- 5/6. Write own slot, then synchronise ------------------------- */
    /* Passive-target epoch for the whole lifetime of the window. */
    MPI_CHECK(MPI_Win_lock_all(MPI_MODE_NOCHECK, win));

    *slots[shm_rank] = shm_rank;

    /* Publish my store, wait for everyone, then make others' stores visible
     * (MPI-3 unified memory model pattern). */
    MPI_CHECK(MPI_Win_sync(win));
    MPI_CHECK(MPI_Barrier(shm_comm));
    MPI_CHECK(MPI_Win_sync(win));

    /* ---- 7/8. Read every slot and verify ------------------------------- */
    for (r = 0; r < shm_size; r++) {
        values[r] = *slots[r];
        if (values[r] != r) {
            local_errors++;
        }
    }

    /* Print one rank at a time to keep output readable. */
    for (turn = 0; turn < shm_size; turn++) {
        if (turn == shm_rank) {
            printf("[world rank %d | shm rank %d] reads:", g_world_rank, shm_rank);
            for (r = 0; r < shm_size; r++) {
                printf(" slot[%d]=%d%s", r, values[r],
                       (values[r] == r) ? "" : "(WRONG)");
            }
            printf("  -> %s\n", local_errors ? "FAIL" : "ok");
            fflush(stdout);
        }
        MPI_CHECK(MPI_Barrier(shm_comm));
    }

    MPI_CHECK(MPI_Allreduce(&local_errors, &global_errors, 1, MPI_INT,
                            MPI_SUM, MPI_COMM_WORLD));
    if (g_world_rank == 0) {
        if (global_errors == 0) {
            printf("RESULT: PASS - all ranks read correct values from shared memory\n");
        } else {
            printf("RESULT: FAIL - %d incorrect value(s) observed\n", global_errors);
        }
        fflush(stdout);
    }

    /* ---- 9. Clean up --------------------------------------------------- */
    MPI_CHECK(MPI_Win_unlock_all(win));
    MPI_CHECK(MPI_Win_free(&win));
    MPI_CHECK(MPI_Comm_free(&shm_comm));
    free((void *)slots);
    free(values);

    MPI_Finalize();
    return (global_errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
