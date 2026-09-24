# The AWS DiskANN benchmarks were measuring EC2 network bandwidth

2026-09-23, AWS 3-node (Scylla `i4i.xlarge` `--smp 4`, VS `r7i.xlarge`, VDB
client, eu-north-1c), branch `aws-benchmarks` @ `bfa06b4`.

## Result

The VS node is pinned at its sustained network allowance and actively shaped:

    VS     (r7i.xlarge)  rx 186 MiB/s = 1.56 Gbps   <- baseline is 1.5625 Gbps
                         bw_in_allowance_exceeded  +278686 / 10 s  (total 94.2M)
    Scylla (i4i.xlarge)  bw_out_allowance_exceeded  102.8M

Read from `ethtool -S <iface>`. Each vector row is ~6.1 KB on the wire
(measured: 186 MiB/s / 31.3k reads/s), so baseline allows **~32k reads/s** -
the last cell measured 31.3k.

## What this explains

- **Client-side read latency 2.5-24 ms against Scylla's 0.33 ms.** Hypervisor
  packet shaping, not the driver, tokio, locks or permits.
- **Scylla idle at 20-26% reactor utilisation.** It answers fine; the bytes
  cannot cross the wire.
- **Why 16x the fetch permits bought nothing** (48 -> 768 permits, 44.5k ->
  43.7k reads/s). The constraint is bytes/s. Extra concurrency only deepens the
  queue, so latency rises in exact proportion and throughput does not move.
  Every "the pool is saturated" reading was Little's law restating itself.
- **The knee is burst-credit exhaustion, not a row count.** It fires after a
  fixed number of *bytes*. That is why the knee moved between runs (count ~140k
  on the first build of the day, then ~13.2k, then ~10k): each run started with
  a more depleted bucket. The first run after the instances were started held
  its fast phase for ~850 s - a burst duration, not a graph-size effect.
- **"Everything suddenly got slow"**, on both 2026-09-22 and 2026-09-23.

## What this voids

**The whole AWS thread sweep, both days.** 44.5k / 73.9k / 43.7k reads/s at
threads 16 / 64 / 256 is 2.2 / 3.6 / 1.9 Gbps - those cells measured how much
burst credit remained when each started. The 2026-09-22 series
(17 / 37 / 42 / 49.4 QPS at threads 4 / 32 / 64 / 256) is suspect for the same
reason, and the "fetch permit pool is sized off vCPU count" conclusion rests
on it.

Also void: everything in this file's earlier revisions - the permit-pool
ceiling, the client-vs-server latency gap as a software defect, and the
oversubscription model fitted to the sweep.

## Still standing

Measured independently of bandwidth, or unaffected by it:

- Reads per query ~3000 at 50k; reads per insert 850-1500 and growing with
  graph size. Ratios, not rates.
- Edges in ScyllaDB cost ~5% of reads (vectors outnumber them ~20:1).
- Beam width 8 = 2.7x throughput for 1.4 points of recall@10 (laptop, loopback).
- Scylla's own ceiling on `--smp 4` is ~230k reads/s; it was never approached.
- `worker::in_flow` pins at 64/64: adds do run concurrently.
- No lock contention in either graph store (`eu-stack`: 64 PARK, 0 LOCK).

## Consequences for the benchmark

**Bytes per operation is the metric, not QPS.** At ~6.1 KB per vector row and
1100-3000 reads per operation, each query or insert moves 7-18 MB. To reach
Scylla's 230k reads/s you need 230k x 6.1 KB = 1.4 GB/s = **11.2 Gbps**, i.e.
8xlarge-class baseline on *both* sides.

Options, in order of cost:

1. **Co-locate VS with Scylla**, or run both locally. Loopback has no ENA
   allowance. This is also a real product question: if vector-store is meant to
   run beside the node owning the data, the network term largely disappears.
2. **Use the 768-dim dataset** instead of 1536. Halves every byte figure.
3. **Reduce bytes**: beam width, vector cache, PQ. These pay at any instance
   size and are now the highest-value work.
4. **Bigger instances**: r7i 2xl/4xl/8xl = 3.125 / 6.25 / 12.5 Gbps, and the
   Scylla side needs it too.

## Method

- Add `bw_in_allowance_exceeded` to `scylla-monitor.sh` beside reactor
  utilisation. Any cell where it increments is a bandwidth measurement.
- **Measure at baseline, not during burst.** Burst numbers are not
  reproducible: they depend on how long the instance idled beforehand. Either
  drain the bucket with a warm-up and then measure, or run long enough that
  burst is a small fraction of the window.
- A stopped/started instance resumes with a full bucket, which is why the first
  run of a session always looks best.

## Repro

    ethtool -S $(ip -o -4 route show to default | awk '{print $5}') | grep allowance
    cat /sys/class/net/<iface>/statistics/rx_bytes   # sample twice, 10 s apart

Instrumented build (VS node only, uncommitted): `#[hotpath::measure]` on the
DiskANN insert path plus a `HotpathGuardBuilder` in `main()`; metrics server on
port 6770 (`/functions_timing`, `/debug`). `#[hotpath::measure]` reports
nothing without the guard, and `main.rs` has no `#[hotpath::main]`.
