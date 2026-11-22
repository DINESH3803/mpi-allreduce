/*
 * MPI Latency Benchmark with RDMA Verbs Eager & Rendezvous Protocols
 * ==================================================================
 * NHPC Assignment 3
 *
 * This file contains three latency measurement modes:
 *   1. Original MPI_Send/MPI_Recv (baseline)
 *   2. Eager protocol via RDMA verbs (IBV_WR_SEND with 128 KB chunking)
 *   3. Rendezvous protocol via RDMA verbs (IBV_WR_RDMA_WRITE_WITH_IMM)
 *
 * Compile:
 *   mpicc -o mpi-latency mpi-latency.c -libverbs -Wall -O2
 *
 * Run:
 *   mpirun -np 2 ./mpi-latency
 *   mpirun -np 2 ./mpi-latency 0 1        # explicit src/dst ranks
 *
 * Output:
 *   Redirect stdout to capture all three benchmark results:
 *   mpirun -np 2 ./mpi-latency > output.txt 2>&1
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>

/* ─── Constants ────────────────────────────────────────────────────── */

#define MAX_SIZE       (1024 * 1024 * 4)   /* 4 MB                    */
#define ITERATIONS     100
#define WARMUP         10
#define EAGER_BUF_SIZE (128 * 1024)        /* 128 KB pre-alloc buffer */
#define CQ_DEPTH       64
#define MIN_RDMA_SIZE  4096                /* RDMA benchmarks start at 4 KB */

/* ─── Benchmark mode ───────────────────────────────────────────────── */

typedef enum {
    MODE_MPI,
    MODE_EAGER,
    MODE_RENDEZVOUS
} bench_mode_t;

/* ─── QP metadata exchanged between ranks during init ──────────────── */

typedef struct {
    uint32_t qpn;
    uint16_t lid;
    uint32_t psn;
    uint64_t rdv_recv_addr;   /* remote dst buffer address (rendezvous) */
    uint32_t rdv_recv_rkey;   /* remote dst buffer rkey    (rendezvous) */
} qp_info_t;

/* ─── RDMA context – holds all verbs resources ─────────────────────── */

typedef struct {
    struct ibv_context    *ctx;
    struct ibv_pd         *pd;
    struct ibv_cq         *cq;
    struct ibv_qp         *qp;
    struct ibv_port_attr   port_attr;
    int                    ib_port;

    /* Eager pre-registered buffers (128 KB each) */
    char                  *eager_send_buf;
    char                  *eager_recv_buf;
    struct ibv_mr         *eager_send_mr;
    struct ibv_mr         *eager_recv_mr;

    /* User-buffer MRs for rendezvous (registered once, full MAX_SIZE) */
    struct ibv_mr         *user_src_mr;
    struct ibv_mr         *user_dst_mr;

    /* Remote side info */
    uint32_t               remote_qpn;
    uint16_t               remote_lid;
    uint32_t               remote_psn;
    uint64_t               remote_rdv_addr;
    uint32_t               remote_rdv_rkey;

    int                    rank;
    int                    peer_rank;
} rdma_context_t;

/* ═══════════════════════════════════════════════════════════════════════
   Helper utilities
   ═══════════════════════════════════════════════════════════════════════ */

/* Busy-poll the CQ until exactly one completion arrives.
   Returns 0 on success, -1 on error. */
static int poll_cq_blocking(struct ibv_cq *cq, struct ibv_wc *wc)
{
    int ne;
    do {
        ne = ibv_poll_cq(cq, 1, wc);
    } while (ne == 0);

    if (ne < 0) {
        fprintf(stderr, "Error: ibv_poll_cq returned %d\n", ne);
        return -1;
    }
    if (wc->status != IBV_WC_SUCCESS) {
        fprintf(stderr, "Error: CQ completion status %d (%s), opcode %d\n",
                wc->status, ibv_wc_status_str(wc->status), wc->opcode);
        return -1;
    }
    return 0;
}

/* Post a receive WR using the eager recv buffer.
   Works for both eager (data lands in buffer) and rendezvous
   (buffer consumed for IMM notification only; data goes via RDMA). */
static int post_recv_wr(rdma_context_t *rctx)
{
    struct ibv_recv_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;

    sge.addr   = (uintptr_t)rctx->eager_recv_buf;
    sge.length = EAGER_BUF_SIZE;
    sge.lkey   = rctx->eager_recv_mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id   = 0;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    return ibv_post_recv(rctx->qp, &wr, &bad_wr);
}

/* ═══════════════════════════════════════════════════════════════════════
   myverbs_init  –  one-time RDMA setup (called right after MPI_Init)
   ═══════════════════════════════════════════════════════════════════════ */

int myverbs_init(rdma_context_t *rctx, int rank, int peer_rank,
                 char *user_src, char *user_dst, int user_buf_size)
{
    struct ibv_device **dev_list;
    int num_devices, ret;

    memset(rctx, 0, sizeof(*rctx));
    rctx->rank      = rank;
    rctx->peer_rank = peer_rank;
    rctx->ib_port   = 1;

    /* ── 1. Open the first available RDMA device ───────────────────── */
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "Error: no RDMA devices found\n");
        return -1;
    }
    rctx->ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (!rctx->ctx) {
        fprintf(stderr, "Error: ibv_open_device failed\n");
        return -1;
    }

    if (ibv_query_port(rctx->ctx, rctx->ib_port, &rctx->port_attr)) {
        fprintf(stderr, "Error: ibv_query_port failed\n");
        return -1;
    }

    /* ── 2. Protection domain ──────────────────────────────────────── */
    rctx->pd = ibv_alloc_pd(rctx->ctx);
    if (!rctx->pd) {
        fprintf(stderr, "Error: ibv_alloc_pd failed\n");
        return -1;
    }

    /* ── 3. Allocate page-aligned eager buffers (128 KB each) ──────── */
    if (posix_memalign((void **)&rctx->eager_send_buf, getpagesize(),
                       EAGER_BUF_SIZE)) {
        fprintf(stderr, "Error: posix_memalign (eager_send) failed\n");
        return -1;
    }
    if (posix_memalign((void **)&rctx->eager_recv_buf, getpagesize(),
                       EAGER_BUF_SIZE)) {
        fprintf(stderr, "Error: posix_memalign (eager_recv) failed\n");
        return -1;
    }
    memset(rctx->eager_send_buf, 0, EAGER_BUF_SIZE);
    memset(rctx->eager_recv_buf, 0, EAGER_BUF_SIZE);

    /* ── 4. Register memory regions ────────────────────────────────── */
    rctx->eager_send_mr = ibv_reg_mr(rctx->pd, rctx->eager_send_buf,
                                     EAGER_BUF_SIZE,
                                     IBV_ACCESS_LOCAL_WRITE);
    if (!rctx->eager_send_mr) {
        fprintf(stderr, "Error: ibv_reg_mr (eager_send) failed\n");
        return -1;
    }

    rctx->eager_recv_mr = ibv_reg_mr(rctx->pd, rctx->eager_recv_buf,
                                     EAGER_BUF_SIZE,
                                     IBV_ACCESS_LOCAL_WRITE |
                                     IBV_ACCESS_REMOTE_WRITE);
    if (!rctx->eager_recv_mr) {
        fprintf(stderr, "Error: ibv_reg_mr (eager_recv) failed\n");
        return -1;
    }

    /* Register user src buffer (local read for RDMA write source) */
    rctx->user_src_mr = ibv_reg_mr(rctx->pd, user_src, user_buf_size,
                                   IBV_ACCESS_LOCAL_WRITE);
    if (!rctx->user_src_mr) {
        fprintf(stderr, "Error: ibv_reg_mr (user_src) failed\n");
        return -1;
    }

    /* Register user dst buffer (remote-writable for rendezvous) */
    rctx->user_dst_mr = ibv_reg_mr(rctx->pd, user_dst, user_buf_size,
                                   IBV_ACCESS_LOCAL_WRITE |
                                   IBV_ACCESS_REMOTE_WRITE);
    if (!rctx->user_dst_mr) {
        fprintf(stderr, "Error: ibv_reg_mr (user_dst) failed\n");
        return -1;
    }

    /* ── 5. Completion queue ───────────────────────────────────────── */
    rctx->cq = ibv_create_cq(rctx->ctx, CQ_DEPTH, NULL, NULL, 0);
    if (!rctx->cq) {
        fprintf(stderr, "Error: ibv_create_cq failed\n");
        return -1;
    }

    /* ── 6. Queue pair (Reliable Connected) ────────────────────────── */
    {
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.send_cq          = rctx->cq;
        qp_attr.recv_cq          = rctx->cq;
        qp_attr.qp_type          = IBV_QPT_RC;
        qp_attr.cap.max_send_wr  = CQ_DEPTH;
        qp_attr.cap.max_recv_wr  = CQ_DEPTH;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;

        rctx->qp = ibv_create_qp(rctx->pd, &qp_attr);
        if (!rctx->qp) {
            fprintf(stderr, "Error: ibv_create_qp failed\n");
            return -1;
        }
    }

    /* ── 7. QP state: RESET → INIT ─────────────────────────────────── */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state        = IBV_QPS_INIT;
        attr.port_num        = rctx->ib_port;
        attr.pkey_index      = 0;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE  |
                               IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ;

        ret = ibv_modify_qp(rctx->qp, &attr,
                            IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                            IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS);
        if (ret) {
            fprintf(stderr, "Error: QP RESET→INIT failed (%d)\n", ret);
            return -1;
        }
    }

    /* ── 8. Exchange QP metadata via MPI ───────────────────────────── */
    {
        qp_info_t local_info, remote_info;
        MPI_Status mpi_status;

        local_info.qpn           = rctx->qp->qp_num;
        local_info.lid           = rctx->port_attr.lid;
        local_info.psn           = 0;
        local_info.rdv_recv_addr = (uintptr_t)user_dst;
        local_info.rdv_recv_rkey = rctx->user_dst_mr->rkey;

        /* Ordered exchange to avoid deadlock */
        if (rank < peer_rank) {
            MPI_Send(&local_info,  sizeof(qp_info_t), MPI_BYTE,
                     peer_rank, 99, MPI_COMM_WORLD);
            MPI_Recv(&remote_info, sizeof(qp_info_t), MPI_BYTE,
                     peer_rank, 99, MPI_COMM_WORLD, &mpi_status);
        } else {
            MPI_Recv(&remote_info, sizeof(qp_info_t), MPI_BYTE,
                     peer_rank, 99, MPI_COMM_WORLD, &mpi_status);
            MPI_Send(&local_info,  sizeof(qp_info_t), MPI_BYTE,
                     peer_rank, 99, MPI_COMM_WORLD);
        }

        rctx->remote_qpn      = remote_info.qpn;
        rctx->remote_lid       = remote_info.lid;
        rctx->remote_psn       = remote_info.psn;
        rctx->remote_rdv_addr  = remote_info.rdv_recv_addr;
        rctx->remote_rdv_rkey  = remote_info.rdv_recv_rkey;
    }

    /* ── 9. QP state: INIT → RTR ──────────────────────────────────── */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state              = IBV_QPS_RTR;
        attr.path_mtu              = IBV_MTU_4096;
        attr.dest_qp_num           = rctx->remote_qpn;
        attr.rq_psn                = rctx->remote_psn;
        attr.max_dest_rd_atomic    = 1;
        attr.min_rnr_timer         = 1;
        attr.ah_attr.is_global     = 0;
        attr.ah_attr.dlid          = rctx->remote_lid;
        attr.ah_attr.sl            = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num      = rctx->ib_port;

        ret = ibv_modify_qp(rctx->qp, &attr,
                            IBV_QP_STATE              | IBV_QP_AV                 |
                            IBV_QP_PATH_MTU           | IBV_QP_DEST_QPN           |
                            IBV_QP_RQ_PSN             | IBV_QP_MAX_DEST_RD_ATOMIC |
                            IBV_QP_MIN_RNR_TIMER);
        if (ret) {
            fprintf(stderr, "Error: QP INIT→RTR failed (%d)\n", ret);
            return -1;
        }
    }

    /* ── 10. QP state: RTR → RTS ──────────────────────────────────── */
    {
        struct ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state      = IBV_QPS_RTS;
        attr.timeout        = 14;
        attr.retry_cnt      = 7;
        attr.rnr_retry      = 7;   /* infinite RNR retry */
        attr.sq_psn         = 0;
        attr.max_rd_atomic  = 1;

        ret = ibv_modify_qp(rctx->qp, &attr,
                            IBV_QP_STATE     | IBV_QP_TIMEOUT         |
                            IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY      |
                            IBV_QP_SQ_PSN    | IBV_QP_MAX_QP_RD_ATOMIC);
        if (ret) {
            fprintf(stderr, "Error: QP RTR→RTS failed (%d)\n", ret);
            return -1;
        }
    }

    /* ── 11. Pre-post one receive WR so the QP is ready ────────────── */
    if (post_recv_wr(rctx)) {
        fprintf(stderr, "Error: initial post_recv_wr failed\n");
        return -1;
    }

    if (rank == 0)
        printf("RDMA verbs initialization successful.\n\n");

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
   myverbs_finalize  –  tear down all RDMA resources
   ═══════════════════════════════════════════════════════════════════════ */

void myverbs_finalize(rdma_context_t *rctx)
{
    if (rctx->qp)            ibv_destroy_qp(rctx->qp);
    if (rctx->cq)            ibv_destroy_cq(rctx->cq);
    if (rctx->user_dst_mr)   ibv_dereg_mr(rctx->user_dst_mr);
    if (rctx->user_src_mr)   ibv_dereg_mr(rctx->user_src_mr);
    if (rctx->eager_recv_mr) ibv_dereg_mr(rctx->eager_recv_mr);
    if (rctx->eager_send_mr) ibv_dereg_mr(rctx->eager_send_mr);
    if (rctx->pd)            ibv_dealloc_pd(rctx->pd);
    if (rctx->ctx)           ibv_close_device(rctx->ctx);
    free(rctx->eager_send_buf);
    free(rctx->eager_recv_buf);
}

/* ═══════════════════════════════════════════════════════════════════════
   TASK 1 — Eager protocol
   Copies user data into a pre-registered 128 KB buffer, sends via
   IBV_WR_SEND.  Large transfers are chunked automatically.
   ═══════════════════════════════════════════════════════════════════════ */

int myverbs_send(rdma_context_t *rctx, void *buf, int len)
{
    int offset, chunk;
    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    struct ibv_wc wc;

    for (offset = 0; offset < len; offset += EAGER_BUF_SIZE) {
        chunk = len - offset;
        if (chunk > EAGER_BUF_SIZE) chunk = EAGER_BUF_SIZE;

        /* Copy user data → pre-registered send buffer */
        memcpy(rctx->eager_send_buf, (char *)buf + offset, chunk);

        sge.addr   = (uintptr_t)rctx->eager_send_buf;
        sge.length = chunk;
        sge.lkey   = rctx->eager_send_mr->lkey;

        memset(&wr, 0, sizeof(wr));
        wr.wr_id      = 0;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.opcode     = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        if (ibv_post_send(rctx->qp, &wr, &bad_wr)) {
            fprintf(stderr, "Error: ibv_post_send (eager) failed\n");
            return -1;
        }
        /* Wait for send completion before next chunk */
        if (poll_cq_blocking(rctx->cq, &wc)) return -1;
    }
    return 0;
}

int myverbs_recv(rdma_context_t *rctx, void *buf, int len)
{
    int offset, chunk;
    struct ibv_wc wc;

    for (offset = 0; offset < len; offset += EAGER_BUF_SIZE) {
        chunk = len - offset;
        if (chunk > EAGER_BUF_SIZE) chunk = EAGER_BUF_SIZE;

        /* A recv WR is already posted (from init or the previous re-post) */
        if (poll_cq_blocking(rctx->cq, &wc)) return -1;

        /* Copy received data → user buffer */
        memcpy((char *)buf + offset, rctx->eager_recv_buf, chunk);

        /* Re-post recv WR for the next incoming chunk / message */
        if (post_recv_wr(rctx)) {
            fprintf(stderr, "Error: re-post recv WR (eager) failed\n");
            return -1;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
   TASK 2 — Rendezvous protocol
   User buffers are pre-registered in myverbs_init.  The sender pushes
   data directly into the receiver's buffer via RDMA Write with Immediate.
   The receiver polls for the IMM completion — zero-copy on the recv side.
   ═══════════════════════════════════════════════════════════════════════ */

int myverbs_send_rdv(rdma_context_t *rctx, void *buf, int len)
{
    struct ibv_send_wr wr, *bad_wr = NULL;
    struct ibv_sge sge;
    struct ibv_wc wc;

    sge.addr   = (uintptr_t)buf;
    sge.length = len;
    sge.lkey   = rctx->user_src_mr->lkey;

    memset(&wr, 0, sizeof(wr));
    wr.wr_id               = 1;
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags          = IBV_SEND_SIGNALED;
    wr.imm_data            = htonl((uint32_t)len);
    wr.wr.rdma.remote_addr = rctx->remote_rdv_addr;
    wr.wr.rdma.rkey        = rctx->remote_rdv_rkey;

    if (ibv_post_send(rctx->qp, &wr, &bad_wr)) {
        fprintf(stderr, "Error: ibv_post_send (rdv RDMA_WRITE_WITH_IMM) failed\n");
        return -1;
    }
    /* Wait for RDMA write completion (implies data delivered to remote) */
    if (poll_cq_blocking(rctx->cq, &wc)) return -1;

    return 0;
}

int myverbs_recv_rdv(rdma_context_t *rctx, void *buf, int len)
{
    struct ibv_wc wc;
    (void)buf;  /* data is already in buf via RDMA — zero copy */
    (void)len;

    /* A recv WR is already posted; poll for the IMM notification */
    if (poll_cq_blocking(rctx->cq, &wc)) return -1;

    /* Re-post recv WR for the next RDMA Write with IMM */
    if (post_recv_wr(rctx)) {
        fprintf(stderr, "Error: re-post recv WR (rdv) failed\n");
        return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
   Benchmark driver — runs warmup + timed iterations for any mode
   ═══════════════════════════════════════════════════════════════════════ */

static void run_benchmark(int rank, int src_rank, int dst_rank,
                          char *src, char *dst,
                          rdma_context_t *rctx,
                          bench_mode_t mode, const char *mode_name,
                          int min_size, int max_size)
{
    int msg_size, i;
    MPI_Status mpi_status;
    struct timeval start, end;
    float total_time;

    /* Header */
    if (rank == src_rank)
        printf("===== %s =====\n%10s   %16s\n",
               mode_name, "Size", "Latency (usecs)");

    /* ── Warmup ────────────────────────────────────────────────────── */
    for (msg_size = min_size; msg_size <= max_size; msg_size *= 2) {
        for (i = 0; i < WARMUP; i++) {
            if (rank == src_rank) {
                switch (mode) {
                case MODE_MPI:
                    MPI_Send(src, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD);
                    MPI_Recv(dst, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &mpi_status);
                    break;
                case MODE_EAGER:
                    myverbs_send(rctx, src, msg_size);
                    myverbs_recv(rctx, dst, msg_size);
                    break;
                case MODE_RENDEZVOUS:
                    myverbs_send_rdv(rctx, src, msg_size);
                    myverbs_recv_rdv(rctx, dst, msg_size);
                    break;
                }
            } else if (rank == dst_rank) {
                switch (mode) {
                case MODE_MPI:
                    MPI_Recv(dst, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &mpi_status);
                    MPI_Send(src, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD);
                    break;
                case MODE_EAGER:
                    myverbs_recv(rctx, dst, msg_size);
                    myverbs_send(rctx, src, msg_size);
                    break;
                case MODE_RENDEZVOUS:
                    myverbs_recv_rdv(rctx, dst, msg_size);
                    myverbs_send_rdv(rctx, src, msg_size);
                    break;
                }
            }
        }
    }

    /* ── Timed benchmark ───────────────────────────────────────────── */
    for (msg_size = min_size; msg_size <= max_size; msg_size *= 2) {
        MPI_Barrier(MPI_COMM_WORLD);

        gettimeofday(&start, NULL);
        for (i = 0; i < ITERATIONS; i++) {
            if (rank == src_rank) {
                switch (mode) {
                case MODE_MPI:
                    MPI_Send(src, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD);
                    MPI_Recv(dst, msg_size, MPI_CHAR, dst_rank, 1, MPI_COMM_WORLD, &mpi_status);
                    break;
                case MODE_EAGER:
                    myverbs_send(rctx, src, msg_size);
                    myverbs_recv(rctx, dst, msg_size);
                    break;
                case MODE_RENDEZVOUS:
                    myverbs_send_rdv(rctx, src, msg_size);
                    myverbs_recv_rdv(rctx, dst, msg_size);
                    break;
                }
            } else if (rank == dst_rank) {
                switch (mode) {
                case MODE_MPI:
                    MPI_Recv(dst, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &mpi_status);
                    MPI_Send(src, msg_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD);
                    break;
                case MODE_EAGER:
                    myverbs_recv(rctx, dst, msg_size);
                    myverbs_send(rctx, src, msg_size);
                    break;
                case MODE_RENDEZVOUS:
                    myverbs_recv_rdv(rctx, dst, msg_size);
                    myverbs_send_rdv(rctx, src, msg_size);
                    break;
                }
            }
        }
        gettimeofday(&end, NULL);

        /* Half-RTT averaged over iterations (same formula as original) */
        total_time = ((end.tv_sec - start.tv_sec) * 1000000.0 +
                      (end.tv_usec - start.tv_usec)) / 2.0;
        total_time /= ITERATIONS;

        if (rank == src_rank)
            printf("%10d   %16.2f\n", msg_size, total_time);
    }

    if (rank == src_rank)
        printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════
   main
   ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    int size, rank, src_rank = 0, dst_rank = 1;
    char *src = NULL, *dst = NULL;
    rdma_context_t rctx;

    if (argc > 1) src_rank = atoi(argv[1]);
    if (argc > 2) dst_rank = atoi(argv[2]);

    /* Allocate page-aligned buffers (required for RDMA registration) */
    if (posix_memalign((void **)&src, getpagesize(), MAX_SIZE) ||
        posix_memalign((void **)&dst, getpagesize(), MAX_SIZE)) {
        perror("cannot allocate page-aligned memory");
        exit(1);
    }
    memset(src, 'A', MAX_SIZE);
    memset(dst, 0,   MAX_SIZE);

    /* ── MPI init ──────────────────────────────────────────────────── */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == src_rank)
        printf("Latency benchmark between src_rank=%d and dst_rank=%d\n\n",
               src_rank, dst_rank);

    /* ── RDMA init (sets up QP, registers memory, exchanges metadata) */
    if (myverbs_init(&rctx, rank,
                     (rank == src_rank) ? dst_rank : src_rank,
                     src, dst, MAX_SIZE)) {
        fprintf(stderr, "Rank %d: myverbs_init failed — aborting\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ──────────────────────────────────────────────────────────────── */
    /* Benchmark 1: Original MPI (1 B → 4 MB)                         */
    /* ──────────────────────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    run_benchmark(rank, src_rank, dst_rank, src, dst, &rctx,
                  MODE_MPI, "MPI Latency Benchmark (MPI_Send/MPI_Recv)",
                  1, MAX_SIZE);

    /* ──────────────────────────────────────────────────────────────── */
    /* Benchmark 2: Eager protocol (4 KB → 4 MB)                      */
    /* ──────────────────────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    run_benchmark(rank, src_rank, dst_rank, src, dst, &rctx,
                  MODE_EAGER, "Eager Protocol (RDMA Verbs IBV_WR_SEND)",
                  MIN_RDMA_SIZE, MAX_SIZE);

    /* ──────────────────────────────────────────────────────────────── */
    /* Benchmark 3: Rendezvous protocol (4 KB → 4 MB)                 */
    /* ──────────────────────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    run_benchmark(rank, src_rank, dst_rank, src, dst, &rctx,
                  MODE_RENDEZVOUS, "Rendezvous Protocol (RDMA_WRITE_WITH_IMM)",
                  MIN_RDMA_SIZE, MAX_SIZE);

    /* ── Cleanup ───────────────────────────────────────────────────── */
    myverbs_finalize(&rctx);
    free(src);
    free(dst);
    MPI_Finalize();
    return 0;
}
