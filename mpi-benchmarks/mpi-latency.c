/*
 * myverbs-latency.c  --  Eager vs Rendezvous protocols over RDMA verbs
 *
 * Replaces MPI_Send / MPI_Recv in the MPI latency benchmark with
 * myverbs_send / myverbs_recv and measures one-way latency for:
 *
 *   eager        copy user data -> preregistered bounce buffer -> SEND;
 *                receiver copies bounce buffer -> user buffer.
 *                Messages > EAGER_SIZE are split into EAGER_SIZE chunks.
 *   rndv         rendezvous, memory registered on EVERY transfer:
 *                receiver registers its buffer, sends {addr,rkey,len} to
 *                the sender (channel semantics), sender does
 *                RDMA WRITE WITH IMMEDIATE straight into the user buffer.
 *   rndv_prereg  same protocol, but user buffers were registered once up
 *                front (isolates the protocol cost from registration cost).
 *
 * Build:  mpicc -O2 -Wall myverbs-latency.c -o myverbs-latency -libverbs
 * Run  :  mpirun -genv I_MPI_FABRICS=shm -np 2 ./myverbs-latency
 *         (shm: MPI is only used for launching, QP-info exchange, barriers)
 *
 * Optional: -DMIN_SIZE=1 to start from 1 byte instead of 4KB.
 * Env     : MYVERBS_DEV=<hca name>, MYVERBS_GID_IDX=<n> (RoCE only)
 */
#include <mpi.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

#ifndef MIN_SIZE
#define MIN_SIZE   4096
#endif
#define MAX_SIZE   (1024*1024*4)
#define WARMUP     10          /* untimed ping-pongs per (mode,size)        */
#define ITERATIONS 100         /* ping-pongs per timed trial                */
#define TRIALS     5           /* timed trials; we report the MEDIAN trial  */

#define EAGER_SIZE (128*1024)  /* one preallocated eager buffer             */
#define NSLOTS     (MAX_SIZE / EAGER_SIZE)  /* preposted recv buffers: enough for a
                                  whole max-size message, so the sender can never
                                  outrun the receiver (no flow control needed)  */
#define NSEND      4           /* send bounce buffers (pipelining)          */
#define PORT_NUM   1

#define DIE(...) do { fprintf(stderr, "[myverbs] " __VA_ARGS__); \
                      fprintf(stderr, "\n"); MPI_Abort(MPI_COMM_WORLD, 1); } while (0)

enum { MODE_EAGER = 0, MODE_RNDV, MODE_RNDV_PRE, NMODES };
static const char *mode_name[NMODES] = { "eager", "rndv", "rndv_prereg" };
static int g_mode = MODE_EAGER;

struct conn_info { uint16_t lid; uint32_t qpn; uint32_t psn; union ibv_gid gid; };
struct rndv_ctrl { uint64_t addr; uint32_t rkey; uint32_t len; };   /* the "handle" */
struct region    { char *addr; size_t len; struct ibv_mr *mr; };

static struct {
    struct ibv_context  *ctx;
    struct ibv_pd       *pd;
    struct ibv_cq       *scq, *rcq;        /* separate send / recv completion queues */
    struct ibv_qp       *qp;
    struct ibv_port_attr port;
    int                  is_roce, gid_idx;
    char                *pool;             /* one page-aligned block, registered once */
    struct ibv_mr       *pool_mr;
    char                *send_buf[NSEND];
    char                *recv_buf[NSLOTS];
    struct region        pre[4];           /* user buffers registered up front */
    int                  npre;
} V;

static double now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

/* ====================== low-level verbs helpers ======================== */

/* Busy-poll one completion; abort on error status or after 30 s (so a bug
 * produces an error message instead of a silent hang). */
static struct ibv_wc poll_blocking(struct ibv_cq *cq, const char *what)
{
    struct ibv_wc wc;
    unsigned long spins = 0;
    double t0 = 0;
    for (;;) {
        int n = ibv_poll_cq(cq, 1, &wc);
        if (n < 0) DIE("ibv_poll_cq failed (%s)", what);
        if (n > 0) break;
        if ((++spins & 0xfffff) == 0) {
            if (t0 == 0) t0 = now_us();
            else if (now_us() - t0 > 30e6) DIE("timeout (30s) waiting for: %s", what);
        }
    }
    if (wc.status != IBV_WC_SUCCESS)
        DIE("%s: completion error: %s (%d)", what, ibv_wc_status_str(wc.status), wc.status);
    return wc;
}

static void post_recv_slot(int i)     /* give receive buffer i to the NIC */
{
    struct ibv_sge sge;
    struct ibv_recv_wr wr, *bad = NULL;
    memset(&sge, 0, sizeof sge);
    memset(&wr, 0, sizeof wr);
    sge.addr = (uintptr_t)V.recv_buf[i]; sge.length = EAGER_SIZE; sge.lkey = V.pool_mr->lkey;
    wr.wr_id = i; wr.sg_list = &sge; wr.num_sge = 1;
    if (ibv_post_recv(V.qp, &wr, &bad)) DIE("ibv_post_recv failed (slot %d)", i);
}

static void post_send_buf(void *addr, uint32_t len, uint32_t lkey)   /* plain SEND */
{
    struct ibv_sge sge;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&sge, 0, sizeof sge);
    memset(&wr, 0, sizeof wr);
    sge.addr = (uintptr_t)addr; sge.length = len; sge.lkey = lkey;
    wr.opcode = IBV_WR_SEND; wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge; wr.num_sge = 1;
    if (ibv_post_send(V.qp, &wr, &bad)) DIE("ibv_post_send (SEND) failed");
}

static void post_rdma_write_imm(void *addr, uint32_t len, uint32_t lkey,
                                uint64_t raddr, uint32_t rkey, uint32_t imm)
{
    struct ibv_sge sge;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&sge, 0, sizeof sge);
    memset(&wr, 0, sizeof wr);
    sge.addr = (uintptr_t)addr; sge.length = len; sge.lkey = lkey;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(imm);
    wr.sg_list = &sge; wr.num_sge = 1;
    wr.wr.rdma.remote_addr = raddr; wr.wr.rdma.rkey = rkey;
    if (ibv_post_send(V.qp, &wr, &bad)) DIE("ibv_post_send (RDMA_WRITE_IMM) failed");
}

/* QP state machine: RESET -> INIT -> RTR -> RTS */
static void qp_to_init(void)
{
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = PORT_NUM;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(V.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
        DIE("modify QP -> INIT failed");
}
static void qp_to_rtr(const struct conn_info *r)
{
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTR; a.path_mtu = V.port.active_mtu;
    a.dest_qp_num = r->qpn; a.rq_psn = r->psn;
    a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    a.ah_attr.dlid = r->lid; a.ah_attr.sl = 0; a.ah_attr.src_path_bits = 0;
    a.ah_attr.port_num = PORT_NUM;
    if (V.is_roce) {
        a.ah_attr.is_global = 1; a.ah_attr.grh.dgid = r->gid;
        a.ah_attr.grh.sgid_index = V.gid_idx; a.ah_attr.grh.hop_limit = 1;
    }
    if (ibv_modify_qp(V.qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
        DIE("modify QP -> RTR failed");
}
static void qp_to_rts(uint32_t my_psn)
{
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7;
    a.sq_psn = my_psn; a.max_rd_atomic = 1;
    if (ibv_modify_qp(V.qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
        DIE("modify QP -> RTS failed");
}

/* ============================ myverbs_init ============================= */

static void myverbs_init(int rank)
{
    int peer = 1 - rank, n = 0, i;

    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list || n == 0) DIE("no RDMA devices found");
    const char *want = getenv("MYVERBS_DEV");
    struct ibv_device *dev = NULL;
    for (i = 0; i < n; i++)
        if (!want || !strcmp(ibv_get_device_name(list[i]), want)) { dev = list[i]; break; }
    if (!dev) DIE("requested device not found");
    char devname[64];
    snprintf(devname, sizeof devname, "%s", ibv_get_device_name(dev));
    V.ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (!V.ctx) DIE("ibv_open_device failed");

    V.pd = ibv_alloc_pd(V.ctx);
    if (!V.pd) DIE("ibv_alloc_pd failed");
    if (ibv_query_port(V.ctx, PORT_NUM, &V.port)) DIE("ibv_query_port failed");
    if (V.port.state != IBV_PORT_ACTIVE) DIE("port %d is not ACTIVE", PORT_NUM);
    V.is_roce = (V.port.link_layer == IBV_LINK_LAYER_ETHERNET);
    V.gid_idx = getenv("MYVERBS_GID_IDX") ? atoi(getenv("MYVERBS_GID_IDX")) : 0;

    V.scq = ibv_create_cq(V.ctx, 256, NULL, NULL, 0);
    V.rcq = ibv_create_cq(V.ctx, 256, NULL, NULL, 0);
    if (!V.scq || !V.rcq) DIE("ibv_create_cq failed");

    struct ibv_qp_init_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.send_cq = V.scq; qa.recv_cq = V.rcq; qa.qp_type = IBV_QPT_RC;
    qa.sq_sig_all = 1;
    qa.cap.max_send_wr = 64; qa.cap.max_recv_wr = 64;
    qa.cap.max_send_sge = 1; qa.cap.max_recv_sge = 1;
    V.qp = ibv_create_qp(V.pd, &qa);
    if (!V.qp) DIE("ibv_create_qp failed");

    /* preallocated, page-aligned, pre-registered eager buffers */
    size_t pool_size = (size_t)(NSEND + NSLOTS) * EAGER_SIZE;
    if (posix_memalign((void **)&V.pool, 4096, pool_size)) DIE("posix_memalign failed");
    memset(V.pool, 0, pool_size);
    V.pool_mr = ibv_reg_mr(V.pd, V.pool, pool_size, IBV_ACCESS_LOCAL_WRITE);
    if (!V.pool_mr) DIE("ibv_reg_mr (eager pool) failed -- check 'ulimit -l'");
    for (i = 0; i < NSEND;  i++) V.send_buf[i] = V.pool + (size_t)i * EAGER_SIZE;
    for (i = 0; i < NSLOTS; i++) V.recv_buf[i] = V.pool + (size_t)(NSEND + i) * EAGER_SIZE;

    /* swap QP info with the peer over MPI */
    struct conn_info me, them;
    memset(&me, 0, sizeof me);
    me.lid = V.port.lid; me.qpn = V.qp->qp_num;
    srand48(getpid() + rank); me.psn = lrand48() & 0xffffff;
    if (V.is_roce && ibv_query_gid(V.ctx, PORT_NUM, V.gid_idx, &me.gid)) DIE("ibv_query_gid failed");
    MPI_Sendrecv(&me, sizeof me, MPI_BYTE, peer, 99, &them, sizeof them, MPI_BYTE, peer, 99,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    qp_to_init();
    for (i = 0; i < NSLOTS; i++) post_recv_slot(i);   /* prepost BEFORE RTR */
    qp_to_rtr(&them);
    qp_to_rts(me.psn);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("[rank %d] verbs ready: dev=%s link=%s lid=%u qpn=0x%x peer_qpn=0x%x mtu=%d\n",
           rank, devname, V.is_roce ? "Ethernet(RoCE)" : "InfiniBand",
           me.lid, me.qpn, them.qpn, 128 << V.port.active_mtu);
    fflush(stdout);
}

/* Register a user buffer once, up front (used by mode rndv_prereg). */
static void myverbs_prereg(void *p, size_t len)
{
    if (V.npre >= 4) DIE("too many preregistered regions");
    struct ibv_mr *mr = ibv_reg_mr(V.pd, p, len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) DIE("ibv_reg_mr (user buffer) failed -- check 'ulimit -l'");
    V.pre[V.npre].addr = p; V.pre[V.npre].len = len; V.pre[V.npre].mr = mr;
    V.npre++;
}

static void myverbs_finalize(void)
{
    int i;
    for (i = 0; i < V.npre; i++) ibv_dereg_mr(V.pre[i].mr);
    if (V.qp)      ibv_destroy_qp(V.qp);
    if (V.scq)     ibv_destroy_cq(V.scq);
    if (V.rcq)     ibv_destroy_cq(V.rcq);
    if (V.pool_mr) ibv_dereg_mr(V.pool_mr);
    free(V.pool);
    if (V.pd)      ibv_dealloc_pd(V.pd);
    if (V.ctx)     ibv_close_device(V.ctx);
}

/* ===================== TASK 1: EAGER protocol ========================== */

static void eager_send(const char *buf, size_t len)
{
    if (len > (size_t)NSLOTS * EAGER_SIZE) DIE("message too large for eager ring");
    size_t off = 0;
    int s = 0, outstanding = 0;
    while (off < len) {
        size_t n = len - off < EAGER_SIZE ? len - off : EAGER_SIZE;
        if (outstanding == NSEND) {              /* reuse a bounce buffer only after */
            poll_blocking(V.scq, "eager send");  /* its send has completed (in order) */
            outstanding--;
        }
        memcpy(V.send_buf[s], buf + off, n);     /* user buffer -> preallocated buffer */
        post_send_buf(V.send_buf[s], (uint32_t)n, V.pool_mr->lkey);
        outstanding++; s = (s + 1) % NSEND; off += n;
    }
    while (outstanding) { poll_blocking(V.scq, "eager send drain"); outstanding--; }
}

static void eager_recv(char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        struct ibv_wc wc = poll_blocking(V.rcq, "eager recv");
        if (wc.opcode != IBV_WC_RECV) DIE("eager recv: unexpected opcode %d", wc.opcode);
        int slot = (int)wc.wr_id;
        size_t n = wc.byte_len;
        if (off + n > len) DIE("eager recv: message longer than receive buffer");
        memcpy(buf + off, V.recv_buf[slot], n);  /* preallocated buffer -> user buffer */
        off += n;
        post_recv_slot(slot);                    /* hand the buffer back */
    }
}

/* ==================== TASK 2: RENDEZVOUS protocol ====================== */

/* Memory registration: in rndv_prereg mode reuse the up-front registration;
 * otherwise register now (and deregister after the transfer). */
static struct ibv_mr *get_mr(void *p, size_t len, int *temporary)
{
    int i;
    if (g_mode == MODE_RNDV_PRE) {
        for (i = 0; i < V.npre; i++)
            if ((char *)p >= V.pre[i].addr && (char *)p + len <= V.pre[i].addr + V.pre[i].len) {
                *temporary = 0;
                return V.pre[i].mr;
            }
    }
    struct ibv_mr *mr = ibv_reg_mr(V.pd, p, len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) DIE("ibv_reg_mr failed in rendezvous");
    *temporary = 1;
    return mr;
}

/* Receiver: register user buffer, send handle, wait for the RDMA write. */
static void rndv_recv(void *buf, size_t len)
{
    int temp;
    struct ibv_mr *mr = get_mr(buf, len, &temp);              /* 1. register first      */

    struct rndv_ctrl c = { (uint64_t)(uintptr_t)buf, mr->rkey, (uint32_t)len };
    memcpy(V.send_buf[0], &c, sizeof c);
    post_send_buf(V.send_buf[0], sizeof c, V.pool_mr->lkey);  /* 2. send {addr,rkey,len} */
    poll_blocking(V.scq, "rndv ctrl send");

    struct ibv_wc wc = poll_blocking(V.rcq, "rndv wait for RDMA write");   /* 3. data lands */
    if (wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) DIE("rndv recv: unexpected opcode %d", wc.opcode);
    if (ntohl(wc.imm_data) != (uint32_t)len)     DIE("rndv recv: immediate length mismatch");
    post_recv_slot((int)wc.wr_id);                            /* the imm consumed a recv */
    if (temp) ibv_dereg_mr(mr);
}

/* Sender: register user buffer, wait for peer's handle, RDMA write with imm. */
static void rndv_send(const void *buf, size_t len)
{
    int temp;
    struct ibv_mr *mr = get_mr((void *)buf, len, &temp);      /* 1. register first      */

    struct ibv_wc wc = poll_blocking(V.rcq, "rndv wait for handle");       /* 2. get handle */
    if (wc.opcode != IBV_WC_RECV || wc.byte_len != sizeof(struct rndv_ctrl))
        DIE("rndv send: bad control message");
    struct rndv_ctrl c;
    memcpy(&c, V.recv_buf[wc.wr_id], sizeof c);
    post_recv_slot((int)wc.wr_id);
    if (c.len != (uint32_t)len) DIE("rndv send: length mismatch (%u vs %zu)", c.len, len);

    post_rdma_write_imm((void *)buf, (uint32_t)len, mr->lkey, c.addr, c.rkey, (uint32_t)len);
    poll_blocking(V.scq, "rndv RDMA write");                  /* 3. zero-copy transfer  */
    if (temp) ibv_dereg_mr(mr);
}

/* ====== the two functions that replace MPI_Send / MPI_Recv ============= */

static int myverbs_send(const void *buf, size_t len, int peer)
{
    (void)peer;                               /* single QP -> single peer */
    if (g_mode == MODE_EAGER) eager_send((const char *)buf, len);
    else                      rndv_send(buf, len);
    return 0;
}
static int myverbs_recv(void *buf, size_t len, int peer)
{
    (void)peer;
    if (g_mode == MODE_EAGER) eager_recv((char *)buf, len);
    else                      rndv_recv(buf, len);
    return 0;
}


/* ===================== original MPI baseline =========================== */

static void mpi_pingpong(int rank, int src_rank, int dst_rank,
                         char *src, char *dst, size_t size, int iters)
{
    for (int i = 0; i < iters; i++) {
        if (rank == src_rank) {
            MPI_Send(src, (int)size, MPI_CHAR, dst_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(dst, (int)size, MPI_CHAR, dst_rank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else if (rank == dst_rank) {
            MPI_Recv(dst, (int)size, MPI_CHAR, src_rank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(src, (int)size, MPI_CHAR, src_rank, 0, MPI_COMM_WORLD);
        }
    }
}

static double mpi_timed_trial(int rank, int src_rank, int dst_rank,
                              char *src, char *dst, size_t size, int iters)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now_us();
    mpi_pingpong(rank, src_rank, dst_rank, src, dst, size, iters);
    double t1 = now_us();
    return ((t1 - t0) / 2.0) / iters;
}

static int cmp_d(const void *a, const void *b);

static double mpi_measure(int rank, int src_rank, int dst_rank,
                          char *src, char *dst, size_t size)
{
    mpi_pingpong(rank, src_rank, dst_rank, src, dst, size, WARMUP);

    double tr[TRIALS];
    for (int t = 0; t < TRIALS; t++)
        tr[t] = mpi_timed_trial(rank, src_rank, dst_rank,
                                src, dst, size, ITERATIONS);

    qsort(tr, TRIALS, sizeof tr[0], cmp_d);
    return tr[TRIALS / 2];
}

/* ===================== measurement code ================================ */

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void pingpong(int rank, int src_rank, int dst_rank, char *src, char *dst,
                     size_t size, int iters)
{
    for (int i = 0; i < iters; i++) {
        if (rank == src_rank) {
            myverbs_send(src, size, dst_rank);
            myverbs_recv(dst, size, dst_rank);
        } else if (rank == dst_rank) {
            myverbs_recv(dst, size, src_rank);
            myverbs_send(src, size, src_rank);
        }
    }
}

/* One-way latency (usec) = round trip / 2, averaged over iters -- same formula as the original. */
static double timed_trial(int rank, int src_rank, int dst_rank, char *src, char *dst,
                          size_t size, int iters)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now_us();
    pingpong(rank, src_rank, dst_rank, src, dst, size, iters);
    double t1 = now_us();
    return ((t1 - t0) / 2.0) / iters;
}

/* One-shot correctness check: every byte must arrive, nothing past the end is touched. */
static void verify(int rank, int src_rank, int dst_rank, char *src, char *dst, size_t size)
{
    int bad = 0, all = 0;
    size_t i;
    if (rank == src_rank) {
        for (i = 0; i < size; i++) src[i] = (char)((i * 131 + size) & 0xff);
        myverbs_send(src, size, dst_rank);
    } else if (rank == dst_rank) {
        memset(dst, 0, size);
        if (size < MAX_SIZE) dst[size] = 0x5A;
        myverbs_recv(dst, size, src_rank);
        for (i = 0; i < size; i++)
            if (dst[i] != (char)((i * 131 + size) & 0xff)) { bad = 1; break; }
        if (size < MAX_SIZE && dst[size] != 0x5A) bad = 1;
    }
    MPI_Allreduce(&bad, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (all) DIE("DATA CHECK FAILED: mode=%s size=%zu", mode_name[g_mode], size);
}

int main(int argc, char **argv)
{
    int nprocs, rank, src_rank = 0, dst_rank = 1;
    char *src = NULL, *dst = NULL;

    if (argc > 1) src_rank = atoi(argv[1]);
    if (argc > 2) dst_rank = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs != 2) DIE("run with exactly 2 ranks (-np 2)");
    if (src_rank == dst_rank || src_rank < 0 || src_rank > 1 ||
        dst_rank < 0 || dst_rank > 1)
        DIE("src_rank/dst_rank must be 0 and 1 in some order");

    if (posix_memalign((void **)&src, 4096, MAX_SIZE) ||
        posix_memalign((void **)&dst, 4096, MAX_SIZE))
        DIE("cannot allocate memory");

    memset(src, 'a', MAX_SIZE);
    memset(dst, 0, MAX_SIZE);

    /*
     * First measure the original MPI implementation.
     * This gives the baseline for the same benchmark.
     */
    size_t sizes[32];
    double mpi_res[32];
    int nsizes = 0;

    for (size_t size = MIN_SIZE; size <= MAX_SIZE; size *= 2) {
        sizes[nsizes] = size;
        mpi_res[nsizes] = mpi_measure(rank, src_rank, dst_rank,
                                      src, dst, size);
        nsizes++;
    }

    /*
     * Initialize verbs immediately after MPI setup, as required.
     * Register user buffers once for the rendezvous protocol.
     */
    myverbs_init(rank);
    myverbs_prereg(src, MAX_SIZE);
    myverbs_prereg(dst, MAX_SIZE);

    if (rank == src_rank) {
        printf("\nEager vs Rendezvous Protocol\n");
        printf("One-way latency in usecs (median of %d trials x %d ping-pongs, %d warmup)\n",
               TRIALS, ITERATIONS, WARMUP);
        printf("Eager: 128 KB preregistered bounce buffers + SEND\n");
        printf("Rendezvous: preregistered user buffers + RDMA WRITE WITH IMMEDIATE\n\n");
        printf("%10s  %12s  %12s  %14s  %s\n",
               "Size(B)", "MPI", "Eager", "Rendezvous", "Best");
        fflush(stdout);
    }

    for (int si = 0; si < nsizes; si++) {
        size_t size = sizes[si];
        double eager = 0.0;
        double rndv_pre = 0.0;

        /*
         * The assignment's two RDMA protocols:
         *   1. Eager
         *   2. Rendezvous with memory registered once up front.
         *
         * The implementation also contains MODE_RNDV (registration on
         * every transfer), but it is not included in the final comparison
         * because it is an extra diagnostic mode.
         */
        for (int m = MODE_EAGER; m <= MODE_RNDV_PRE; m++) {
            g_mode = m;

            verify(rank, src_rank, dst_rank, src, dst, size);

            memset(src, 'a', size);
            MPI_Barrier(MPI_COMM_WORLD);
            pingpong(rank, src_rank, dst_rank, src, dst, size, WARMUP);

            double tr[TRIALS];
            for (int t = 0; t < TRIALS; t++)
                tr[t] = timed_trial(rank, src_rank, dst_rank,
                                    src, dst, size, ITERATIONS);

            qsort(tr, TRIALS, sizeof tr[0], cmp_d);

            if (m == MODE_EAGER)
                eager = tr[TRIALS / 2];
            else if (m == MODE_RNDV_PRE)
                rndv_pre = tr[TRIALS / 2];
        }

        if (rank == src_rank) {
            double best = mpi_res[si];
            const char *best_name = "MPI";

            if (eager < best) {
                best = eager;
                best_name = "Eager";
            }
            if (rndv_pre < best) {
                best = rndv_pre;
                best_name = "Rendezvous";
            }

            printf("%10zu  %12.2f  %12.2f  %14.2f  %s\n",
                   size, mpi_res[si], eager, rndv_pre, best_name);
            fflush(stdout);
        }
    }

    if (rank == src_rank)
        printf("\nAll data-integrity checks passed.\n");

    myverbs_finalize();
    free(src);
    free(dst);
    MPI_Finalize();
    return 0;
}
