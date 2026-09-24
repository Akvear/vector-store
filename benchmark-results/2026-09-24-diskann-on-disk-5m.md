# On-disk DiskANN at 500k and 5M: it is cache residency, not dataset size

2026-09-24, AWS. VS `i4i.xlarge` (4 vCPU, 30 GB RAM, 872 GB local NVMe),
ScyllaDB `i4i.xlarge` `--smp 4`, VDB client, eu-north-1c. Branch
`on-disk-diskann` @ `ee8e5c2`, `VECTOR_STORE_DISKANN_BACKEND=disk`,
`VECTOR_STORE_THREADS=16`, OpenAI 1536-d.

Node file is **8.0 KiB per record** (vector + edges): 3.9 GiB at 500k,
38.1 GiB at 5M. It is a `tempfile_in(DATA_DIR)`, unlinked at creation - it dies
with the process, so a restart costs a full rebuild.

## Headline

**Throughput is set by misses per query, and misses are set by how much of the
query working set fits in cache** - not by residency as a fraction, and not by
index size directly.

    QPS ~= drive_IOPS_ceiling / misses_per_query

predicts every drive-bound cell within ~10%, with the ceiling from fio.

**Sizing rule: a 5M / 38 GiB index needs ~15-17 GiB of RAM (about 40% of the
index) to serve at full speed.** Below that, throughput falls roughly in
proportion to cache; above it, extra RAM buys nothing - the box is CPU-bound at
~165 QPS on four cores.

Note that residency *percentage* is not the invariant. 500k at 47% resident
(1.8 GiB cache) gives 572 misses/query, while 5M at 44% resident (17.2 GiB
cache) gives ~6. Both happen to serve ~170 QPS, but for different reasons - the
500k cell is drive-bound and the 5M cell is CPU-bound. What matters is absolute
cache against the working set, and the working set grows sublinearly with N.

## Query curve, 5M (38.1 GiB index, k=10)

| cache | resident | peak QPS | misses/query | bound by |
|---|---|---|---|---|
| 3.0 GiB | 8% | 65 | 1488 | drive (82%) |
| 7.1 GiB | 18% | 103 | 995 | joint |
| 11.1 GiB | 29% | 137 | 431 | joint |
| 13.3 GiB | 34% | 155 | 167 | joint |
| 17.2 GiB | 44% | 170 | ~6 | CPU (402%) |
| 28.7 GiB | 73% | 163 | ~8 | CPU (401%) |

Ramps were `[1, 8, 16, 32]` concurrency; peak is usually c=8 or c=16.
Between 13.3 and 17.2 GiB the misses collapse ~30x - the working set boundary
is in that gap. Above it the box is CPU-bound at ~165 QPS and more cache is
wasted.

## Query curve, 500k (3.9 GiB index, k=10)

| cache | resident | peak QPS | misses/query |
|---|---|---|---|
| 3906 MiB | 100% | 230 | 0 |
| 1840 MiB | 47% | 170 | 572 |
| 940 MiB | 24% | 75 | 1303 |
| 390 MiB | 10% | 73 | 1333 |

Misses plateau below 24% here: dropping 940 -> 390 MiB changed nothing, so
~1300 candidates per query are cold at any cache size worth provisioning.

## k=100 (5M, 3.0 GiB cache, matched pair)

| k | L | peak QPS | misses/query |
|---|---|---|---|
| 10 | 64 | 65 | 1488 |
| 100 | 100 | 54 | 1750 |

`mod.rs:523` sets `l_value = expansion_search.max(k)`, so k=100 raises L per
query with **no rebuild**. L rose 1.56x but misses only 1.18x - the extra list
slots are filled mostly from nodes the search already fetched. **Top-100 costs
~20% throughput over top-10** in the I/O-bound regime.

## Build

5M: **104 min 31 s**, 5,000,000 rows, **797 rows/s average**. 500k: under
10 minutes at 833 rows/s, fully cached throughout.

The 5M build degrades as the written portion outgrows RAM:

| written | resident-of-written | misses/insert | rows/s |
|---|---|---|---|
| 29 GiB | 99% | 15 | 741 |
| 32 GiB | 90% | 78 | 607 |
| 36 GiB | 80% | 146 | 536 |
| 37 GiB | 77% | 173 | 463 |

Misses roughly double per 10 points of residency lost. It finished
CPU-and-drive jointly bound (CPU ~360%, drive 89%), not collapsing. Build
locality is strong: pruning reads hit recently-inserted nodes, so the build
stayed near-free of I/O until the written portion passed the cache ceiling.

## Drive baseline (fio, idle box)

| cell | result |
|---|---|
| 4k, qd64 x4 | 106,068 IOPS, 414 MB/s |
| 8k, qd64 x4 | **90,609 IOPS, 708 MB/s** |
| 8k, qd1 x1 | 7,766 IOPS, **128 us** |

Real I/O size in the workload is ~6-8 KB (a 6.1 KB vector spans two pages and
is fetched as one request), so the 8k row is the relevant ceiling. The 128 us
figure is what a synchronous page fault costs unloaded; under load `await`
reaches 4-5 ms.

## Predictions for 10-20M on 16 GB RAM

Extrapolated, not measured. Stated so they can be checked against a real run.

**Working set scaling.** Full speed needs ~2-4 GiB of cache at 500k and 15-17
GiB at 5M - roughly 5x for 10x the rows, i.e. the working set grows about
`N^0.7`. On that exponent:

| dataset | index | working set (est) |
|---|---|---|
| 5M | 38 GiB | ~16 GiB (measured, 13-17) |
| 10M | 76 GiB | ~25 GiB |
| 20M | 153 GiB | ~40 GiB |

**With 16 GB of RAM** (anon ~1.5-2 GB at these sizes, so ~13 GiB of usable
cache):

| dataset | cache / working set | misses/query (est) | peak QPS (est) |
|---|---|---|---|
| 5M | 81% | ~200 | ~150 |
| 10M | 52% | ~900 | ~100 |
| 20M | 33% | ~1600 | ~55 |

Misses are interpolated from the measured 5M curve expressed against working
set, then scaled up ~1.2x at 10M and ~1.4x at 20M for the deeper graph. QPS is
`90,600 / misses`, so these assume the same single NVMe and that the drive, not
CPU, binds - true for everything below ~550 misses/query on four cores.

**RAM needed to stay CPU-bound**, which is the more useful framing:

| dataset | RAM for full speed | as % of index |
|---|---|---|
| 5M | ~18 GB | 42% |
| 10M | ~28 GB | 33% |
| 20M | ~44 GB | 26% |

**The required fraction falls as the dataset grows**, because the working set
is sublinear in N. That is the encouraging part of this extrapolation: a 20M
index needs proportionally less RAM than a 5M one.

**Build is the binding practical constraint, not query.** The build degrades
with resident-of-written, and on 16 GB that ratio ends far lower than the 77%
this 5M run finished at. Extrapolating the measured build curve (misses roughly
doubling per 10 points of residency lost):

| dataset | end residency on 16 GB | est. avg rows/s | est. build |
|---|---|---|---|
| 5M | ~34% | ~400 | ~3.5 h |
| 10M | ~17% | ~250 | ~11 h |
| 20M | ~9% | ~120 | ~46 h |

A 20M build on this hardware is a multi-day job. More RAM helps the build far
more than it helps queries, and build throughput scales with cores while query
throughput past the knee does not.

**Confidence.** The working-set exponent is fitted to two dataset sizes, so it
is weakly determined - the 10M figures could be out by 30% and the 20M ones by
more. The query model itself (`QPS = ceiling / misses`) held within 10% across
six cells and is the solid part. The cheapest way to test the exponent is one
10M build, which the table says costs ~11 hours on 16 GB or ~5 on a 64 GB box.

## Method

**Squeeze cache without restarting.** `systemctl set-property --runtime
vs-disk.service MemoryHigh=<n>` retunes the live cgroup. The process never
restarts, so the unlinked node file survives. Run VS under
`systemd-run --unit=vs-disk --uid=ubuntu -p MemoryHigh=... -p MemoryMax=...`.

**Use `MemoryHigh`, never `MemoryMax`** - the former applies reclaim pressure,
the latter OOM-kills mid-run.

**Keep the limit above anon or the cell is junk.** Anon grows with thread count
and dataset: 86 MiB idle at 50k/16 threads, 443 MiB at 50k/64 threads, 920 MiB
at 5M. When anon exceeds the limit there is nothing reclaimable and the process
stalls: one 50k cell logged 201k reclaim events with CPU collapsing to 161%
while resident slid 32% -> 18% mid-run. When anon is well under the limit the
same counter reaches 100k+ harmlessly, because dropping clean file pages is
cheap. Read `memory.pressure`, not the event count.

**Lowering a limit overshoots.** The kernel reclaims far past the new ceiling
(20G limit -> dropped to 9 GiB of a possible 19). Warm up until `resident`
plateaus before measuring, or the early ramp steps measure a cold cache.
Raising a limit causes no reclaim; cache grows only as pages are touched.

**Watch resident-of-*written*, not resident-of-file.** The file is preallocated
to `MAX_POINTS`, so during a build `resident/file` looks reassuringly flat at
74% while the meaningful ratio (cache / bytes written) falls continuously.

**Do not let build I/O leak into query stats.** A `tail`-based window reached
back past the `SERVING` transition and attributed the build's 86k IOPS to the
first query cell, inflating misses/query 60x. Filter log lines by epoch against
a boundary recorded when the cell starts.

**`DATA_DIR` must exist and be writable by the VS user.** `create_dir_all` then
`tempfile_in` both fail on a root-owned `/mnt/data`, `add_index` fails, and the
engine retries once a second forever logging only `INFO creating the index`,
because the real error sits behind `debug!` at `engine.rs:232`/`:243`. Third
time this failure mode has cost time; promoting those two to `warn!` is a
one-line fix.

## Recall

**Lowest recorded across all 5M runs: 93.77%.** Reference points: 98.8%
recall@10 and 95.1% recall@100 at 50k on the laptop, so this is consistent with
ordinary degradation at 100x the rows rather than anything anomalous.

Recall does not vary with cache residency - same index, same L, same beam, only
I/O timing differs between cells. **So the whole throughput ladder above was
measured at constant quality**, which is what makes 65 QPS and 170 QPS
comparable. Material recall variation between squeeze cells would have meant
cache state was leaking into results; it did not.

## Open

Untested: residency between 13.3 and 17.2 GiB (would pin the working set
exactly), datasets above 5M, and build behaviour below 77% residency.
