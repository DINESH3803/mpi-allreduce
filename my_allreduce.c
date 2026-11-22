/*
 * my_allreduce.c
 *
 * Single-node Allreduce (MPI_INT, MPI_SUM) over an MPI-3 shared-memory window,
 * with the memory copies and the reduction parallelised with OpenMP.
 *
 * ----------------------------------------------------------------------------
 * Shared-memory layout: one segment per rank, all in one window.
 *
 *     segment of rank r:
 *     +-----------+------------------------------+----------------------+
 *     | header    | input  (in_stride ints)      | output slice         |
 *     | 16 ints   | copy of rank r's sendbuf     | (out_stride ints)    |
 *     +-----------+------------------------------+----------------------+
 *
 *   header : status / count / call sequence number of rank r's current call.
 *            Every rank reads all headers to check the arguments are
 *            consistent across ranks, without an extra MPI collective.
 *   input  : rank r copies its private sendbuf here (phase 1).
 *   output : rank r writes the fully reduced elements of its own slice here
 *            (phase 2). Every rank later copies all slices into its recvbuf
 *            (phase 3).
 *
 * The window is requested with alloc_shared_noncontig=true, so each segment
 * can be placed on its owner's NUMA node. Owner threads also first-touch it.
 * Both are hints only: the MPI library decides where pages end up.
 *
 * ----------------------------------------------------------------------------
 * Algorithm for one call, with P ranks and count elements:
 *
 *   phase 1  copy-in : my sendbuf -> my shared input segment   (OpenMP memcpy)
 *   sync A           : all inputs and headers are visible to everyone
 *   check            : every rank reads all headers; on any error, all ranks
 *                      take the same error path
 *   phase 2  reduce  : for my slice [start, start+len):
 *                        out_me[k] = sum over p of in_p[start+k]   (OpenMP)
 *   sync B           : all output slices are complete and visible
 *   phase 3  gather  : copy every rank's output slice into my recvbuf (OpenMP)
 *   sync C           : (optional, on by default) nobody starts the next call
 *                      until everyone has finished reading the shared data
 *
 * Slices: base = count / P, rem = count % P. Rank r gets base + (r < rem)
 * elements starting at r*base + min(r, rem). This handles counts that are not
 * divisible by P.
 *
 * ----------------------------------------------------------------------------
 * Synchronisation: MPI-3 "unified" memory model pattern, inside one
 * passive-target epoch (MPI_Win_lock_all at init, MPI_Win_unlock_all at
 * finalize):
 *
 *     MPI_Win_sync(win);    my stores to the window are complete/visible
 *     MPI_Barrier(comm);    every rank has reached this point
 *     MPI_Win_sync(win);    I will observe the other ranks' stores
 *
 * All MPI calls happen on the main thread, outside OpenMP parallel regions.
 * An OpenMP parallel region ends with an implicit barrier, so every thread's
 * stores are complete before the main thread calls MPI_Win_sync
 * (MPI_THREAD_FUNNELED is enough).
 *
 * Written for GCC 4.8.5 (-std=gnu99, OpenMP 3.1) and Intel MPI 2018 Update 2.
 */

#include "my_allreduce.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------------------------------------------------------- tunables */

#define HDR_INTS      16     /* 64-byte header per segment                     */
#define ALIGN_INTS    16     /* keep input/output regions 64-byte aligned      */
#define PAR_MIN_INTS  8192   /* below this many ints (32 KB) skip the OpenMP fork */
#define SUMN_BLOCK    4096   /* block length for the generic-P reduction kernel */

/*
 * Sync C (trailing barrier). Strictly, the next call's sync A already stops
 * any rank from overwriting shared data another rank is still reading:
 *   - inputs are only read in phase 2, which every rank finishes before sync B;
 *   - output slices are only rewritten after the next call's sync A, which a
 *     rank reaches only after finishing its own phase 3;
 *   - headers are only read between sync A and sync B.
 * It is kept on by default for this first version, as the algorithm asks.
 * Build with -DMY_ALLREDUCE_TRAILING_BARRIER=0 to measure without it.
 */
#ifndef MY_ALLREDUCE_TRAILING_BARRIER
#define MY_ALLREDUCE_TRAILING_BARRIER 1
#endif

/* header slots */
#define HDR_STATUS 0
#define HDR_COUNT  1
#define HDR_SEQ    2

/* ------------------------------------------------------------------- state */

typedef struct {
    int       initialized;
    int       locked;       /* inside MPI_Win_lock_all epoch                  */
    MPI_Comm  comm;         /* node-local communicator (owned by the library) */
    MPI_Win   win;          /* shared-memory window                            */
    int       rank;         /* rank in comm                                    */
    int       nranks;       /* size of comm                                    */
    int       user_rank;    /* rank in the user's communicator (for messages)  */
    int       max_count;
    long      in_stride;    /* ints reserved for the input region              */
    long      out_stride;   /* ints reserved for the output slice              */
    int       seq;          /* call sequence number                            */
    int     **hdr;          /* hdr[r] : header of rank r                       */
    int     **in;           /* in[r]  : input region of rank r                 */
    int     **out;          /* out[r] : output slice of rank r                 */
} my_state_t;

static my_state_t S;

static void reset_state(void)
{
    int user_rank = S.user_rank;
    memset(&S, 0, sizeof(S));
    /* MPI_COMM_NULL / MPI_WIN_NULL are not necessarily 0 (Intel MPI: they are not). */
    S.comm      = MPI_COMM_NULL;
    S.win       = MPI_WIN_NULL;
    S.user_rank = user_rank;
}

/* ---------------------------------------------------------- error reporting */

static void report(const char *where, const char *what)
{
    fprintf(stderr, "[my_allreduce rank %d] %s: %s\n", S.user_rank, where, what);
    fflush(stderr);
}

static void report_mpi(const char *where, const char *call, int rc)
{
    char msg[MPI_MAX_ERROR_STRING];
    int  len = 0;

    if (MPI_Error_string(rc, msg, &len) != MPI_SUCCESS)
        snprintf(msg, sizeof(msg), "MPI error code %d", rc);
    fprintf(stderr, "[my_allreduce rank %d] %s: %s failed: %s\n",
            S.user_rank, where, call, msg);
    fflush(stderr);
}

/* Return the MPI error code from the current function. */
#define TRY_MPI(where, call)                                         \
    do {                                                             \
        int rc_ = (call);                                            \
        if (rc_ != MPI_SUCCESS) {                                    \
            report_mpi((where), #call, rc_);                         \
            return rc_;                                              \
        }                                                            \
    } while (0)

/* Used in my_allreduce_init: store the code in rc and jump to cleanup. */
#define TRY_MPI_GOTO(call)                                           \
    do {                                                             \
        rc = (call);                                                 \
        if (rc != MPI_SUCCESS) {                                     \
            report_mpi("my_allreduce_init", #call, rc);              \
            goto fail;                                               \
        }                                                            \
    } while (0)

const char *my_allreduce_strerror(int code)
{
    static char buf[MPI_MAX_ERROR_STRING];
    int len = 0;

    switch (code) {
    case MY_ALLREDUCE_SUCCESS:             return "success";
    case MY_ALLREDUCE_ERR_NOT_INITIALIZED: return "my_allreduce_init has not been called";
    case MY_ALLREDUCE_ERR_ALREADY_INIT:    return "my_allreduce_init was already called";
    case MY_ALLREDUCE_ERR_MPI_NOT_INIT:    return "MPI is not initialised";
    case MY_ALLREDUCE_ERR_THREAD_LEVEL:    return "MPI thread support is below MPI_THREAD_FUNNELED";
    case MY_ALLREDUCE_ERR_NOT_MAIN_THREAD: return "called from a thread other than the MPI main thread";
    case MY_ALLREDUCE_ERR_IN_PARALLEL:     return "called from inside an OpenMP parallel region";
    case MY_ALLREDUCE_ERR_BAD_ARG:         return "invalid argument (count out of range or NULL buffer)";
    case MY_ALLREDUCE_ERR_ARG_MISMATCH:    return "ranks passed different counts or are out of step";
    case MY_ALLREDUCE_ERR_NOT_SINGLE_NODE: return "communicator spans more than one shared-memory node";
    case MY_ALLREDUCE_ERR_NO_MEMORY:       return "out of memory or requested size too large";
    case MY_ALLREDUCE_ERR_REMOTE:          return "another rank reported an error";
    case MY_ALLREDUCE_ERR_BAD_SEGMENT:     return "MPI_Win_shared_query returned an unexpected segment";
    default:
        if (code > 0 && MPI_Error_string(code, buf, &len) == MPI_SUCCESS)
            return buf;
        return "unknown error";
    }
}

/* ------------------------------------------------------------------ helpers */

static long round_up(long n, long a)
{
    return ((n + a - 1) / a) * a;
}

/* Contiguous, balanced share [lo, hi) of n items for the calling OpenMP thread. */
static void thread_range(long n, long *lo, long *hi)
{
#ifdef _OPENMP
    long nt = omp_get_num_threads();
    long t  = omp_get_thread_num();
#else
    long nt = 1, t = 0;
#endif
    long base = n / nt, rem = n % nt;

    *lo = t * base + (t < rem ? t : rem);
    *hi = *lo + base + (t < rem ? 1 : 0);
}

/* Slice [start, start+len) of count elements owned by rank r of P. */
static void slice_bounds(int count, int P, int r, long *start, long *len)
{
    long base = count / P, rem = count % P;

    *start = r * base + (r < rem ? r : rem);
    *len   = base + (r < rem ? 1 : 0);
}

/* Sync A/B/C: make my stores visible, wait for all ranks, observe theirs. */
static int sync_all(const char *where)
{
    TRY_MPI(where, MPI_Win_sync(S.win));
    TRY_MPI(where, MPI_Barrier(S.comm));
    TRY_MPI(where, MPI_Win_sync(S.win));
    return MPI_SUCCESS;
}

/* Fail if called from a non-main thread or inside an OpenMP parallel region. */
static int check_calling_context(const char *where)
{
    int is_main = 0;

#ifdef _OPENMP
    if (omp_in_parallel()) {
        report(where, my_allreduce_strerror(MY_ALLREDUCE_ERR_IN_PARALLEL));
        return MY_ALLREDUCE_ERR_IN_PARALLEL;
    }
#endif
    /* MPI_Is_thread_main may be called from any thread. */
    TRY_MPI(where, MPI_Is_thread_main(&is_main));
    if (!is_main) {
        report(where, my_allreduce_strerror(MY_ALLREDUCE_ERR_NOT_MAIN_THREAD));
        return MY_ALLREDUCE_ERR_NOT_MAIN_THREAD;
    }
    return MY_ALLREDUCE_SUCCESS;
}

/* ----------------------------------------------------- OpenMP data movement */

/* dst[0..n) = src[0..n); each thread copies one contiguous block. */
static void parallel_copy(int *dst, const int *src, long n)
{
    if (n <= 0)
        return;
#pragma omp parallel if (n >= PAR_MIN_INTS)
    {
        long lo, hi;
        thread_range(n, &lo, &hi);
        if (hi > lo)
            memcpy(dst + lo, src + lo, (size_t)(hi - lo) * sizeof(int));
    }
}

/* Zero-fill (first touch) n ints with the same thread partitioning. */
static void parallel_zero(int *dst, long n)
{
    if (n <= 0)
        return;
#pragma omp parallel if (n >= PAR_MIN_INTS)
    {
        long lo, hi;
        thread_range(n, &lo, &hi);
        if (hi > lo)
            memset(dst + lo, 0, (size_t)(hi - lo) * sizeof(int));
    }
}

/*
 * Reduction kernels. They are separate functions with restrict-qualified
 * parameters so GCC 4.8 can vectorise them without runtime alias checks
 * (OpenMP 3.1 has no "omp simd").
 */
static void sum2(int *restrict d, const int *restrict a, const int *restrict b,
                 long n)
{
    long k;
    for (k = 0; k < n; k++)
        d[k] = a[k] + b[k];
}

static void sum4(int *restrict d, const int *restrict a, const int *restrict b,
                 const int *restrict c, const int *restrict e, long n)
{
    long k;
    for (k = 0; k < n; k++)
        d[k] = (a[k] + b[k]) + (c[k] + e[k]);
}

/* Any P: works in cache-sized blocks so d stays in cache across the P passes. */
static void sumN(int *restrict d, int *const *in, int P, long off, long n)
{
    long b0, k, len;
    int  p;

    for (b0 = 0; b0 < n; b0 += SUMN_BLOCK) {
        len = (n - b0 < SUMN_BLOCK) ? (n - b0) : SUMN_BLOCK;
        memcpy(d + b0, in[0] + off + b0, (size_t)len * sizeof(int));
        for (p = 1; p < P; p++) {
            const int *restrict s = in[p] + off + b0;
            int *restrict t = d + b0;
            for (k = 0; k < len; k++)
                t[k] += s[k];
        }
    }
}

/* Phase 2: out_me[k] = sum over p of in_p[start + k], for k in [0, len). */
static void reduce_my_slice(long start, long len)
{
    int *const        dst = S.out[S.rank];
    int *const *const in  = S.in;
    const int         P   = S.nranks;

    if (len <= 0)
        return;
#pragma omp parallel if (len >= PAR_MIN_INTS)
    {
        long lo, hi, n, off;
        thread_range(len, &lo, &hi);
        n   = hi - lo;
        off = start + lo;
        if (n > 0) {
            switch (P) {
            case 1:
                memcpy(dst + lo, in[0] + off, (size_t)n * sizeof(int));
                break;
            case 2:
                sum2(dst + lo, in[0] + off, in[1] + off, n);
                break;
            case 4:
                sum4(dst + lo, in[0] + off, in[1] + off, in[2] + off,
                     in[3] + off, n);
                break;
            default:
                sumN(dst + lo, in, P, off, n);
                break;
            }
        }
    }
}

/*
 * Phase 3: recvbuf[0..count) = all output slices, in rank order.
 * The thread split is over the whole result, not per slice, so all threads
 * get equal work even though the slices live in different segments.
 */
static void gather_result(int *recvbuf, int count)
{
    int *const *const out = S.out;
    const int         P   = S.nranks;

    if (count <= 0)
        return;
#pragma omp parallel if (count >= PAR_MIN_INTS)
    {
        long lo, hi, s, len, a, b;
        int  r;
        thread_range(count, &lo, &hi);
        for (r = 0; r < P; r++) {
            slice_bounds(count, P, r, &s, &len);
            a = (lo > s) ? lo : s;                       /* overlap of [lo,hi) */
            b = (hi < s + len) ? hi : s + len;           /* and [s, s+len)     */
            if (b > a)
                memcpy(recvbuf + a, out[r] + (a - s), (size_t)(b - a) * sizeof(int));
        }
    }
}

/* -------------------------------------------------------- resource release */

static int release_resources(const char *where)
{
    int rc, first = MPI_SUCCESS;

    if (S.locked) {
        rc = MPI_Win_unlock_all(S.win);
        if (rc != MPI_SUCCESS) { report_mpi(where, "MPI_Win_unlock_all", rc); first = rc; }
        S.locked = 0;
    }
    if (S.win != MPI_WIN_NULL) {
        rc = MPI_Win_free(&S.win);      /* collective; also synchronises all ranks */
        if (rc != MPI_SUCCESS) {
            report_mpi(where, "MPI_Win_free", rc);
            if (first == MPI_SUCCESS) first = rc;
        }
    }
    if (S.comm != MPI_COMM_NULL) {
        rc = MPI_Comm_free(&S.comm);
        if (rc != MPI_SUCCESS) {
            report_mpi(where, "MPI_Comm_free", rc);
            if (first == MPI_SUCCESS) first = rc;
        }
    }
    free(S.hdr);                         /* S.in and S.out point into this block */
    reset_state();
    return first;
}

/* ===================================================================== API */

int my_allreduce_init(MPI_Comm comm, int max_count)
{
    static const char *where = "my_allreduce_init";
    int       flag = 0, provided = MPI_THREAD_SINGLE, comm_size = 0;
    int       rc = MPI_SUCCESS, err = MY_ALLREDUCE_SUCCESS;
    int       local[3]  = { 0, 0, 0 };
    int       global[3] = { 0, 0, 0 };
    int       r;
    long      seg_ints = 0;
    MPI_Aint  seg_bytes = 0;
    MPI_Info  info = MPI_INFO_NULL;
    int      *base = NULL;
    char      msg[160];

    /*
     * Local checks, before any collective call. Each of these conditions is
     * the same on every rank (same program, same MPI library), so returning
     * early here cannot leave some ranks waiting in a collective.
     */
    if (S.initialized) {
        report(where, my_allreduce_strerror(MY_ALLREDUCE_ERR_ALREADY_INIT));
        return MY_ALLREDUCE_ERR_ALREADY_INIT;
    }
    reset_state();

    if (MPI_Initialized(&flag) != MPI_SUCCESS || !flag) {
        report(where, my_allreduce_strerror(MY_ALLREDUCE_ERR_MPI_NOT_INIT));
        return MY_ALLREDUCE_ERR_MPI_NOT_INIT;
    }
    TRY_MPI(where, MPI_Comm_rank(comm, &S.user_rank));
    TRY_MPI(where, MPI_Comm_size(comm, &comm_size));

    /* MPI must be initialised by the caller with at least FUNNELED. */
    TRY_MPI(where, MPI_Query_thread(&provided));
    if (provided < MPI_THREAD_FUNNELED) {
        snprintf(msg, sizeof(msg),
                 "MPI provides thread level %d, need >= MPI_THREAD_FUNNELED (%d)",
                 provided, MPI_THREAD_FUNNELED);
        report(where, msg);
        return MY_ALLREDUCE_ERR_THREAD_LEVEL;
    }
    rc = check_calling_context(where);
    if (rc != MY_ALLREDUCE_SUCCESS)
        return rc;

    /* ---- collective: node-local communicator -------------------------- */
    TRY_MPI_GOTO(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0 /* keep order */,
                                     MPI_INFO_NULL, &S.comm));
    TRY_MPI_GOTO(MPI_Comm_set_errhandler(S.comm, MPI_ERRORS_RETURN));
    TRY_MPI_GOTO(MPI_Comm_rank(S.comm, &S.rank));
    TRY_MPI_GOTO(MPI_Comm_size(S.comm, &S.nranks));

    /* ---- sizes --------------------------------------------------------- */
    if (max_count <= 0) {
        err = MY_ALLREDUCE_ERR_BAD_ARG;
        report(where, "max_count must be > 0");
    } else if (S.nranks != comm_size) {
        err = MY_ALLREDUCE_ERR_NOT_SINGLE_NODE;
        snprintf(msg, sizeof(msg),
                 "communicator has %d ranks but only %d share this node",
                 comm_size, S.nranks);
        report(where, msg);
    } else {
        S.max_count  = max_count;
        S.in_stride  = round_up((long)max_count, ALIGN_INTS);
        S.out_stride = round_up((long)max_count / S.nranks
                                + ((max_count % S.nranks) ? 1 : 0), ALIGN_INTS);
        seg_ints     = HDR_INTS + S.in_stride + S.out_stride;
        /* MPI_Aint is a signed long on LP64 Linux; check the size fits. */
        if (seg_ints > LONG_MAX / (long)sizeof(int)) {
            err = MY_ALLREDUCE_ERR_NO_MEMORY;
            report(where, "requested segment size overflows MPI_Aint");
        } else {
            seg_bytes = (MPI_Aint)seg_ints * (MPI_Aint)sizeof(int);
        }
    }

    /*
     * Agreement 1 (over the user communicator): did any rank fail, and did
     * every rank pass the same max_count? max(-x) == -min(x).
     */
    local[0] = (err != MY_ALLREDUCE_SUCCESS);
    local[1] = (max_count > 0) ?  max_count : 0;
    local[2] = (max_count > 0) ? -max_count : 0;
    TRY_MPI_GOTO(MPI_Allreduce(local, global, 3, MPI_INT, MPI_MAX, comm));
    if (global[0]) {
        if (err == MY_ALLREDUCE_SUCCESS) {
            err = MY_ALLREDUCE_ERR_REMOTE;
            report(where, "another rank failed during initialisation");
        }
        goto fail;
    }
    if (global[1] != -global[2]) {
        err = MY_ALLREDUCE_ERR_ARG_MISMATCH;
        report(where, "ranks passed different max_count values");
        goto fail;
    }

    /* ---- collective: allocate the shared window ------------------------ */
    TRY_MPI_GOTO(MPI_Info_create(&info));
    TRY_MPI_GOTO(MPI_Info_set(info, "alloc_shared_noncontig", "true"));
    TRY_MPI_GOTO(MPI_Win_allocate_shared(seg_bytes, (int)sizeof(int), info,
                                         S.comm, &base, &S.win));
    TRY_MPI_GOTO(MPI_Info_free(&info));
    TRY_MPI_GOTO(MPI_Win_set_errhandler(S.win, MPI_ERRORS_RETURN));

    /* ---- local: pointers to every rank's segment ----------------------- */
    S.hdr = (int **)malloc(3 * (size_t)S.nranks * sizeof(int *));
    if (S.hdr == NULL) {
        err = MY_ALLREDUCE_ERR_NO_MEMORY;
        report(where, "cannot allocate segment pointer table");
    } else {
        S.in  = S.hdr + S.nranks;
        S.out = S.in  + S.nranks;
        for (r = 0; r < S.nranks; r++) {
            MPI_Aint qsize = 0;
            int      qdisp = 0;
            int     *qbase = NULL;

            rc = MPI_Win_shared_query(S.win, r, &qsize, &qdisp, &qbase);
            if (rc != MPI_SUCCESS) {
                report_mpi(where, "MPI_Win_shared_query", rc);
                err = rc;
                rc  = MPI_SUCCESS;
                break;
            }
            if (qbase == NULL || qsize < seg_bytes || qdisp != (int)sizeof(int)) {
                snprintf(msg, sizeof(msg),
                         "segment of rank %d: base=%p size=%ld disp_unit=%d (expected size>=%ld)",
                         r, (void *)qbase, (long)qsize, qdisp, (long)seg_bytes);
                report(where, msg);
                err = MY_ALLREDUCE_ERR_BAD_SEGMENT;
                break;
            }
            S.hdr[r] = qbase;
            S.in[r]  = qbase + HDR_INTS;
            S.out[r] = qbase + HDR_INTS + S.in_stride;
        }
    }

    /* Agreement 2 (node communicator): did every rank get valid pointers? */
    local[0] = (err != MY_ALLREDUCE_SUCCESS);
    TRY_MPI_GOTO(MPI_Allreduce(local, global, 1, MPI_INT, MPI_MAX, S.comm));
    if (global[0]) {
        if (err == MY_ALLREDUCE_SUCCESS) {
            err = MY_ALLREDUCE_ERR_REMOTE;
            report(where, "another rank failed to map the shared window");
        }
        goto fail;
    }

    /*
     * One passive-target epoch for the window's whole lifetime. MODE_NOCHECK:
     * no rank takes a conflicting exclusive lock. From here on, ordering is
     * provided by the MPI_Win_sync + MPI_Barrier pattern in sync_all().
     */
    TRY_MPI_GOTO(MPI_Win_lock_all(MPI_MODE_NOCHECK, S.win));
    S.locked = 1;

    /* First touch: this rank's threads zero this rank's segment. */
    parallel_zero(S.hdr[S.rank], seg_ints);

    /* Everyone has finished setting up before anyone uses the window. */
    rc = sync_all(where);
    if (rc != MPI_SUCCESS)
        goto fail;

    S.seq         = 0;
    S.initialized = 1;
    return MY_ALLREDUCE_SUCCESS;

fail:
    /*
     * Reached on all ranks together after a failed agreement, or after an
     * MPI error (which with Intel MPI is normally fatal anyway).
     */
    if (info != MPI_INFO_NULL)
        MPI_Info_free(&info);
    release_resources(where);
    return (rc != MPI_SUCCESS) ? rc : err;
}

int my_mpi_allreduce(const int *sendbuf, int *recvbuf, int count)
{
    static const char *where = "my_mpi_allreduce";
    int   local_status = MY_ALLREDUCE_SUCCESS;
    int   global_status = MY_ALLREDUCE_SUCCESS;
    int   r, rc;
    int  *my_hdr;
    long  my_start, my_len;
    char  msg[160];

    /* Same on every rank, and no collective has been called yet. */
    if (!S.initialized) {
        report(where, my_allreduce_strerror(MY_ALLREDUCE_ERR_NOT_INITIALIZED));
        return MY_ALLREDUCE_ERR_NOT_INITIALIZED;
    }
    rc = check_calling_context(where);
    if (rc != MY_ALLREDUCE_SUCCESS)
        return rc;

    /*
     * Check arguments, but do NOT return early: a bad argument on one rank
     * must not leave the others stuck in a barrier. The status goes into
     * this rank's header, and all ranks decide together after sync A.
     */
    if (count < 0 || count > S.max_count) {
        local_status = MY_ALLREDUCE_ERR_BAD_ARG;
        snprintf(msg, sizeof(msg), "count %d outside [0, %d]", count, S.max_count);
        report(where, msg);
    } else if (count > 0 && (sendbuf == NULL || recvbuf == NULL)) {
        local_status = MY_ALLREDUCE_ERR_BAD_ARG;
        report(where, "NULL send or receive buffer");
    }

    S.seq++;
    my_hdr = S.hdr[S.rank];
    my_hdr[HDR_STATUS] = local_status;
    my_hdr[HDR_COUNT]  = count;
    my_hdr[HDR_SEQ]    = S.seq;

    /* ---- phase 1: copy my private sendbuf into my shared input ---------- */
    if (local_status == MY_ALLREDUCE_SUCCESS)
        parallel_copy(S.in[S.rank], sendbuf, count);

    /* ---- sync A: every input segment and header is ready and visible ---- */
    rc = sync_all(where);
    if (rc != MPI_SUCCESS)
        return rc;

    /*
     * All ranks read the same headers, so they all reach the same verdict
     * and follow the same path below.
     */
    for (r = 0; r < S.nranks; r++) {
        const int *h = S.hdr[r];
        if (h[HDR_STATUS] != MY_ALLREDUCE_SUCCESS) {
            global_status = MY_ALLREDUCE_ERR_REMOTE;
        } else if (h[HDR_COUNT] != count || h[HDR_SEQ] != S.seq) {
            if (global_status == MY_ALLREDUCE_SUCCESS)
                global_status = MY_ALLREDUCE_ERR_ARG_MISMATCH;
        }
    }
    if (global_status != MY_ALLREDUCE_SUCCESS) {
        if (local_status == MY_ALLREDUCE_SUCCESS) {
            report(where, my_allreduce_strerror(global_status));
        }
        /* Wait until every rank has read the headers before anyone returns
         * and rewrites its header in a later call. */
        rc = sync_all(where);
        if (rc != MPI_SUCCESS)
            return rc;
        return (local_status != MY_ALLREDUCE_SUCCESS) ? local_status : global_status;
    }

    /* ---- phase 2: reduce my slice from every rank's input --------------- */
    slice_bounds(count, S.nranks, S.rank, &my_start, &my_len);
    reduce_my_slice(my_start, my_len);

    /* ---- sync B: every output slice is complete and visible ------------- */
    rc = sync_all(where);
    if (rc != MPI_SUCCESS)
        return rc;

    /* ---- phase 3: copy the complete result into my private recvbuf ------ */
    gather_result(recvbuf, count);

#if MY_ALLREDUCE_TRAILING_BARRIER
    /* ---- sync C: everyone has finished reading the shared buffers ------- */
    rc = sync_all(where);
    if (rc != MPI_SUCCESS)
        return rc;
#endif

    return MY_ALLREDUCE_SUCCESS;
}

int my_allreduce_finalize(void)
{
    static const char *where = "my_allreduce_finalize";
    int rc;

    if (!S.initialized) {
        report(where, my_allreduce_strerror(MY_ALLREDUCE_ERR_NOT_INITIALIZED));
        return MY_ALLREDUCE_ERR_NOT_INITIALIZED;
    }
    rc = check_calling_context(where);
    if (rc != MY_ALLREDUCE_SUCCESS)
        return rc;

    /* Close the epoch, free the window (collective), free the communicator. */
    return release_resources(where);
}
