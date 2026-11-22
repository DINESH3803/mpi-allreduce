/*
 * mpi-allreduce-shm-bench.c
 *
 * Compares the library MPI_Allreduce with my_mpi_allreduce (shared memory +
 * OpenMP) for MPI_INT / MPI_SUM, message sizes 1, 2, 4, 8, 16, 32 and 64 MB.
 *
 * For each size:
 *   1. Correctness (untimed): fill the two destination buffers with different
 *      sentinel values, run both implementations once, then
 *        - memcmp(dst_std, dst_my) on every rank;
 *        - check dst_std against the closed-form expected sum, so a broken
 *          or intercepted MPI_Allreduce (e.g. sharedlibrary.c preloaded)
 *          cannot make both results "agree" on garbage.
 *   2. Timing: warmup calls, then timed calls. Before each timed call there is
 *      an MPI_Barrier; each rank times its own call with MPI_Wtime. For every
 *      iteration the maximum time over ranks is taken (MPI_MAX), then the
 *      avg / min / max of those maxima are reported.
 *   3. Post-timing check: memcmp the last results of both implementations again.
 *
 * Output: lines starting with '#' are comments. Everything else is CSV.
 *
 * Usage:  mpirun -np <ranks> ./mpi-allreduce-shm-bench [iters] [warmup]
 *         defaults: iters = 20, warmup = 5
 *
 * Written for GCC 4.8.5 (-std=gnu99, OpenMP 3.1) and Intel MPI 2018 Update 2.
 */

#include <mpi.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "my_allreduce.h"

#define MIB              (1024L * 1024L)
#define MIN_BYTES        (1L * MIB)
#define MAX_BYTES        (64L * MIB)
#define DEFAULT_ITERS    20
#define DEFAULT_WARMUP   5
#define MAX_ITERS        100000
#define BUF_ALIGN        64

/* Input values lie in [0, VALUE_MOD). A sum over P ranks stays below
 * P * VALUE_MOD, which fits in a signed int for P <= MAX_RANKS_NO_OVERFLOW. */
#define VALUE_MOD              1000003L
#define MAX_RANKS_NO_OVERFLOW  ((int)(INT_MAX / VALUE_MOD))

/* Sentinels: results are never negative, so these can never look valid. */
#define SENTINEL_STD  (-1)
#define SENTINEL_MY   (-2)

/* Per-rank verification flags, combined across ranks with MPI_BOR. */
#define FLAG_MY_ERROR       1   /* my_mpi_allreduce returned an error        */
#define FLAG_MISMATCH       2   /* memcmp(dst_std, dst_my) != 0              */
#define FLAG_STD_WRONG      4   /* MPI_Allreduce result != expected          */
#define FLAG_POST_MISMATCH  8   /* results differ after the timing loops     */
#define FLAG_STD_TIME_ERR  16   /* MPI_Allreduce failed during timing        */
#define FLAG_MY_TIME_ERR   32   /* my_mpi_allreduce failed during timing     */

#define MAX_PRINT_MISMATCH  5

static int g_rank = -1;

/* ------------------------------------------------------------ error helpers */

static void die_mpi(int rc, const char *call, const char *file, int line)
{
    char msg[MPI_MAX_ERROR_STRING];
    int  len = 0;

    if (MPI_Error_string(rc, msg, &len) != MPI_SUCCESS)
        snprintf(msg, sizeof(msg), "MPI error code %d", rc);
    fprintf(stderr, "[rank %d] FATAL: %s failed at %s:%d: %s\n",
            g_rank, call, file, line, msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define CHECK_MPI(call)                                              \
    do {                                                             \
        int rc_ = (call);                                            \
        if (rc_ != MPI_SUCCESS)                                      \
            die_mpi(rc_, #call, __FILE__, __LINE__);                 \
    } while (0)

static void die(const char *msg)
{
    fprintf(stderr, "[rank %d] FATAL: %s\n", g_rank, msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

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

/* ------------------------------------------------------------ data helpers */

/* Depends on both rank and index, so a misplaced or missing contribution
 * changes the sum. Always in [0, VALUE_MOD). */
static int gen_value(int rank, long i)
{
    return (int)((i * 7L + (long)rank * 1009L) % VALUE_MOD);
}

static int *alloc_ints(long n)
{
    void *p = NULL;

    if (n <= 0 || (unsigned long)n > SIZE_MAX / sizeof(int))
        return NULL;
    if (posix_memalign(&p, BUF_ALIGN, (size_t)n * sizeof(int)) != 0)
        return NULL;
    return (int *)p;
}

/* Parallel fills also first-touch the pages from this rank's threads. */
static void fill_input(int *buf, long n, int rank)
{
    long i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++)
        buf[i] = gen_value(rank, i);
}

static void fill_const(int *buf, long n, int value)
{
    long i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < n; i++)
        buf[i] = value;
}

/* Number of elements where a != b. *first gets the first such index (or -1). */
static long count_diff(const int *a, const int *b, long n, long *first)
{
    long i, bad = 0, lowest = LONG_MAX;
#pragma omp parallel for schedule(static) reduction(+:bad) reduction(min:lowest)
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            bad++;
            if (i < lowest)
                lowest = i;
        }
    }
    *first = bad ? lowest : -1;
    return bad;
}

static long expected_at(long i, int nranks)
{
    long e = 0;
    int  r;
    for (r = 0; r < nranks; r++)
        e += gen_value(r, i);
    return e;
}

/* Compare buf against the closed-form sum over all ranks. */
static long check_expected(const int *buf, long n, int nranks, long *first)
{
    long i, bad = 0, lowest = LONG_MAX;
#pragma omp parallel for schedule(static) reduction(+:bad) reduction(min:lowest)
    for (i = 0; i < n; i++) {
        if ((long)buf[i] != expected_at(i, nranks)) {
            bad++;
            if (i < lowest)
                lowest = i;
        }
    }
    *first = bad ? lowest : -1;
    return bad;
}

/* Print the first few differing elements (stderr, this rank only). */
static void print_mismatches(const char *what, long bytes, const int *std,
                             const int *my, long n, int nranks)
{
    long i, shown = 0;

    for (i = 0; i < n && shown < MAX_PRINT_MISMATCH; i++) {
        if (std[i] != my[i]) {
            fprintf(stderr,
                    "[rank %d]   %s bytes=%ld index=%ld std=%d my=%d expected=%ld\n",
                    g_rank, what, bytes, i, std[i], my[i], expected_at(i, nranks));
            shown++;
        }
    }
}

/* ------------------------------------------------------------------ timing */

typedef int (*allreduce_fn)(const int *, int *, int);

static int call_std(const int *s, int *r, int n)
{
    return MPI_Allreduce(s, r, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

static int call_my(const int *s, int *r, int n)
{
    return my_mpi_allreduce(s, r, n);
}

typedef struct {
    double avg_us, min_us, max_us;
    int    failed;               /* any call on any rank returned an error */
} timing_t;

/*
 * Warmup calls, then iters timed calls with a barrier before each one.
 * t_local and t_max hold iters doubles each. All ranks make the same calls.
 */
static timing_t time_impl(allreduce_fn fn, const int *src, int *dst, int count,
                          int warmup, int iters, double *t_local, double *t_max)
{
    timing_t res;
    int      i, failed = 0, any_failed = 0;
    double   t0, sum = 0.0;

    for (i = 0; i < warmup; i++)
        if (fn(src, dst, count) != MPI_SUCCESS)
            failed = 1;

    for (i = 0; i < iters; i++) {
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        t0 = MPI_Wtime();
        if (fn(src, dst, count) != MPI_SUCCESS)
            failed = 1;
        t_local[i] = MPI_Wtime() - t0;
    }

    /* Per-iteration maximum over ranks: a collective finishes when its
     * slowest rank finishes. */
    CHECK_MPI(MPI_Allreduce(t_local, t_max, iters, MPI_DOUBLE, MPI_MAX,
                            MPI_COMM_WORLD));
    CHECK_MPI(MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX,
                            MPI_COMM_WORLD));

    res.min_us = res.max_us = t_max[0] * 1e6;
    for (i = 0; i < iters; i++) {
        double us = t_max[i] * 1e6;
        sum += us;
        if (us < res.min_us) res.min_us = us;
        if (us > res.max_us) res.max_us = us;
    }
    res.avg_us = sum / iters;
    res.failed = any_failed;
    return res;
}

/* --------------------------------------------------------------- argument */

static int parse_int(const char *s, int lo, int hi, int *out)
{
    char *end = NULL;
    long  v = strtol(s, &end, 10);

    if (end == s || *end != '\0' || v < lo || v > hi)
        return 0;
    *out = (int)v;
    return 1;
}

static void status_string(int flags, char *buf, size_t len)
{
    if (flags == 0) {
        snprintf(buf, len, "PASS");
        return;
    }
    snprintf(buf, len, "FAIL%s%s%s%s%s%s",
             (flags & FLAG_MY_ERROR)      ? ":my_error"       : "",
             (flags & FLAG_MISMATCH)      ? ":mismatch"       : "",
             (flags & FLAG_STD_WRONG)     ? ":std_wrong"      : "",
             (flags & FLAG_POST_MISMATCH) ? ":post_mismatch"  : "",
             (flags & FLAG_STD_TIME_ERR)  ? ":std_time_error" : "",
             (flags & FLAG_MY_TIME_ERR)   ? ":my_time_error"  : "");
}

/* ===================================================================== main */

int main(int argc, char **argv)
{
    int     provided = MPI_THREAD_SINGLE, nranks = 0, rc, i;
    int     iters = DEFAULT_ITERS, warmup = DEFAULT_WARMUP;
    int     max_count, threads, thr_local[2], thr_global[2];
    int     any_failure = 0;
    long    bytes;
    int    *src = NULL, *dst_std = NULL, *dst_my = NULL;
    double *t_local = NULL, *t_max = NULL;
    char    host[MPI_MAX_PROCESSOR_NAME];
    char   *all_hosts = NULL;
    char    libver[MPI_MAX_LIBRARY_VERSION_STRING];
    int     len = 0;

    /* ---- MPI with FUNNELED: only the main thread calls MPI ---------------- */
    rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "FATAL: MPI_Init_thread failed (code %d)\n", rc);
        return EXIT_FAILURE;
    }
    CHECK_MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    if (provided < MPI_THREAD_FUNNELED) {
        if (g_rank == 0)
            fprintf(stderr, "FATAL: MPI provides %s; MPI_THREAD_FUNNELED is required "
                    "(with Intel MPI, try linking with -mt_mpi)\n",
                    thread_level_name(provided));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- arguments (same on every rank, so the decision is consistent) -- */
    if ((argc > 1 && !parse_int(argv[1], 1, MAX_ITERS, &iters)) ||
        (argc > 2 && !parse_int(argv[2], 0, MAX_ITERS, &warmup)) || argc > 3) {
        if (g_rank == 0)
            fprintf(stderr, "usage: %s [iters 1..%d] [warmup 0..%d]\n",
                    argv[0], MAX_ITERS, MAX_ITERS);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (nranks > MAX_RANKS_NO_OVERFLOW) {
        if (g_rank == 0)
            fprintf(stderr, "FATAL: %d ranks would overflow the int sums (max %d)\n",
                    nranks, MAX_RANKS_NO_OVERFLOW);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- buffers: separate destinations for the two implementations ----- */
    max_count = (int)(MAX_BYTES / (long)sizeof(int));
    src       = alloc_ints(max_count);
    dst_std   = alloc_ints(max_count);
    dst_my    = alloc_ints(max_count);
    t_local   = (double *)malloc((size_t)iters * sizeof(double));
    t_max     = (double *)malloc((size_t)iters * sizeof(double));
    if (!src || !dst_std || !dst_my || !t_local || !t_max)
        die("cannot allocate benchmark buffers");   /* MPI_Abort stops all ranks */

    fill_input(src, max_count, g_rank);
    fill_const(dst_std, max_count, SENTINEL_STD);
    fill_const(dst_my,  max_count, SENTINEL_MY);

    /* ---- custom allreduce setup (collective) ---------------------------- */
    rc = my_allreduce_init(MPI_COMM_WORLD, max_count);
    if (rc != MY_ALLREDUCE_SUCCESS) {
        fprintf(stderr, "[rank %d] FATAL: my_allreduce_init: %s\n",
                g_rank, my_allreduce_strerror(rc));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- run description ------------------------------------------------- */
#ifdef _OPENMP
    threads = omp_get_max_threads();
#else
    threads = 1;
#endif
    thr_local[0] = threads;
    thr_local[1] = -threads;
    CHECK_MPI(MPI_Allreduce(thr_local, thr_global, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

    memset(host, 0, sizeof(host));
    CHECK_MPI(MPI_Get_processor_name(host, &len));
    if (g_rank == 0) {
        all_hosts = (char *)malloc((size_t)nranks * MPI_MAX_PROCESSOR_NAME);
        if (all_hosts == NULL)
            die("cannot allocate host name table");
    }
    CHECK_MPI(MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         all_hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         0, MPI_COMM_WORLD));

    if (g_rank == 0) {
        memset(libver, 0, sizeof(libver));
        if (MPI_Get_library_version(libver, &len) == MPI_SUCCESS) {
            for (i = 0; libver[i] != '\0'; i++)
                if (libver[i] == '\n' || libver[i] == '\r')
                    libver[i] = ' ';
        } else {
            snprintf(libver, sizeof(libver), "unknown");
        }
        printf("# MPI_Allreduce vs my_mpi_allreduce (MPI_INT, MPI_SUM)\n");
        printf("# MPI library: %s\n", libver);
        printf("# thread level provided: %s\n", thread_level_name(provided));
        printf("# ranks: %d, OpenMP threads per rank: %d", nranks, thr_global[0]);
        if (-thr_global[1] != thr_global[0])
            printf(" (min %d)", -thr_global[1]);
        printf("\n# warmup: %d, timed iterations: %d\n", warmup, iters);
        for (i = 0; i < nranks; i++)
            printf("# rank %d on %s\n", i, all_hosts + (size_t)i * MPI_MAX_PROCESSOR_NAME);
        printf("# time = per-call time, max over ranks per iteration; "
               "avg/min/max over iterations; microseconds\n");
        printf("# speedup = std_avg_us / my_avg_us; GBps = bytes / avg time / 1e9\n");
        printf("ranks,threads,bytes,count,std_avg_us,std_min_us,std_max_us,"
               "my_avg_us,my_min_us,my_max_us,speedup,std_GBps,my_GBps,status\n");
        fflush(stdout);
    }

    /* ---- size sweep: 1, 2, 4, ..., 64 MB --------------------------------- */
    for (bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes *= 2) {
        int      count = (int)(bytes / (long)sizeof(int));
        int      flags = 0, gflags = 0, nfail = 0, gnfail = 0;
        long     nbad, first;
        timing_t ts, tm;
        char     status[128];

        /* 1. correctness, untimed ------------------------------------------- */
        fill_const(dst_std, count, SENTINEL_STD);
        fill_const(dst_my,  count, SENTINEL_MY);

        CHECK_MPI(MPI_Allreduce(src, dst_std, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
        rc = my_mpi_allreduce(src, dst_my, count);

        if (rc != MY_ALLREDUCE_SUCCESS) {
            flags |= FLAG_MY_ERROR;
            fprintf(stderr, "[rank %d] ERROR: my_mpi_allreduce(bytes=%ld) returned %d: %s\n",
                    g_rank, bytes, rc, my_allreduce_strerror(rc));
        } else if (memcmp(dst_std, dst_my, (size_t)bytes) != 0) {
            flags |= FLAG_MISMATCH;
            nbad = count_diff(dst_std, dst_my, count, &first);
            fprintf(stderr, "[rank %d] MISMATCH: bytes=%ld: %ld of %d elements differ, first at index %ld\n",
                    g_rank, bytes, nbad, count, first);
            print_mismatches("mismatch", bytes, dst_std, dst_my, count, nranks);
        }

        nbad = check_expected(dst_std, count, nranks, &first);
        if (nbad != 0) {
            flags |= FLAG_STD_WRONG;
            fprintf(stderr, "[rank %d] ERROR: MPI_Allreduce result wrong at %ld of %d elements "
                    "(first index %ld: got %d, expected %ld)\n",
                    g_rank, nbad, count, first, dst_std[first], expected_at(first, nranks));
        }

        /* 2. timing ---------------------------------------------------------- */
        ts = time_impl(call_std, src, dst_std, count, warmup, iters, t_local, t_max);
        if (ts.failed)
            flags |= FLAG_STD_TIME_ERR;

        /* Skip timing the custom version if it already failed on any rank
         * (decided collectively, so every rank takes the same branch). */
        CHECK_MPI(MPI_Allreduce(&flags, &gflags, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD));
        if (gflags & FLAG_MY_ERROR) {
            tm.avg_us = tm.min_us = tm.max_us = 0.0;
            tm.failed = 1;
        } else {
            tm = time_impl(call_my, src, dst_my, count, warmup, iters, t_local, t_max);
            if (tm.failed)
                flags |= FLAG_MY_TIME_ERR;

            /* 3. the last results of both must still match ------------------- */
            if (!tm.failed && !ts.failed &&
                memcmp(dst_std, dst_my, (size_t)bytes) != 0) {
                flags |= FLAG_POST_MISMATCH;
                nbad = count_diff(dst_std, dst_my, count, &first);
                fprintf(stderr, "[rank %d] MISMATCH after timing: bytes=%ld: %ld elements differ, first at %ld\n",
                        g_rank, bytes, nbad, first);
                print_mismatches("post-timing", bytes, dst_std, dst_my, count, nranks);
            }
        }

        /* Combine verdicts from all ranks. */
        nfail = (flags != 0);
        CHECK_MPI(MPI_Allreduce(&flags, &gflags, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Allreduce(&nfail, &gnfail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
        if (gflags != 0)
            any_failure = 1;

        if (g_rank == 0) {
            status_string(gflags, status, sizeof(status));
            if (gflags != 0)
                printf("# VERIFY bytes=%ld: FAILED on %d of %d ranks (%s)\n",
                       bytes, gnfail, nranks, status);
            if (gflags & FLAG_MY_ERROR) {
                printf("%d,%d,%ld,%d,%.2f,%.2f,%.2f,nan,nan,nan,nan,%.3f,nan,%s\n",
                       nranks, thr_global[0], bytes, count,
                       ts.avg_us, ts.min_us, ts.max_us,
                       (double)bytes / (ts.avg_us * 1e3), status);
            } else {
                printf("%d,%d,%ld,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%s\n",
                       nranks, thr_global[0], bytes, count,
                       ts.avg_us, ts.min_us, ts.max_us,
                       tm.avg_us, tm.min_us, tm.max_us,
                       ts.avg_us / tm.avg_us,
                       (double)bytes / (ts.avg_us * 1e3),     /* bytes/us/1e3 = GB/s */
                       (double)bytes / (tm.avg_us * 1e3),
                       status);
            }
            fflush(stdout);
        }
    }

    if (g_rank == 0) {
        printf("# overall: %s\n", any_failure ? "FAILURES DETECTED (see stderr)" : "all sizes PASS");
        fflush(stdout);
    }

    /* ---- cleanup -------------------------------------------------------- */
    rc = my_allreduce_finalize();
    if (rc != MY_ALLREDUCE_SUCCESS) {
        fprintf(stderr, "[rank %d] my_allreduce_finalize: %s\n",
                g_rank, my_allreduce_strerror(rc));
        any_failure = 1;
    }

    free(all_hosts);
    free(t_max);
    free(t_local);
    free(dst_my);
    free(dst_std);
    free(src);

    MPI_Finalize();
    return any_failure ? EXIT_FAILURE : EXIT_SUCCESS;
}
