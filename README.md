What is this?
=============
Python script to optimize block sizes for blis. Requires a modified BLIS source that allows setting block sizes dynamically via environment variables.


Usage:
------

```
usage: blistune.py [-h] -t THREADS [-o {gemm,hemm,herk,trmm,trsm}]
                   [-P {fp32,fp64}] [-s LAYOUT] [-p PROBLEM_SIZE] [-r REPEATS]
                   [--aggregate {max,mean,min}] [-n N_TRIALS] [--seed SEED]
                   [--timeout TIMEOUT] [-c CPU_MASK] [--omp-places OMP_PLACES]
                   [--omp-proc-bind OMP_PROC_BIND] [-e KEY=VALUE] [-v]
                   [--save-best SAVE_BEST] [--quiet] [--mr-nr MRxNR]
                   [--jc-nt-max N] [--ic-nt-max N] [--jr-nt-max N]
                   [--ir-nt-max N] [--kc-min KC_MIN] [--kc-max KC_MAX]
                   [--kc-step KC_STEP] [--kc-headroom KC_HEADROOM]
                   [--mr-in-mc-min MR_IN_MC_MIN] [--mr-in-mc-max MR_IN_MC_MAX]
                   [--mr-in-mc-step MR_IN_MC_STEP]
                   [--mr-in-mc-headroom MR_IN_MC_HEADROOM]
                   [--nr-in-nc-min NR_IN_NC_MIN] [--nr-in-nc-max NR_IN_NC_MAX]
                   [--nr-in-nc-step NR_IN_NC_STEP]
                   [--nr-in-nc-headroom NR_IN_NC_HEADROOM]
                   blis_build_dir

Autotune BLIS block sizes and thread decomposition with Optuna.

positional arguments:
  blis_build_dir        BLIS build directory (drivers are looked up under
                        <dir>/test/3)

options:
  -h, --help            show this help message and exit
  -t, --threads THREADS
                        total thread count; the JC/IC/JR/IR product must equal
                        this
  -o, --operation {gemm,hemm,herk,trmm,trsm}
                        which test_<op>_blis_mt.x driver to benchmark
                        (default: gemm)
  -P, --precision {fp32,fp64}
                        datatype passed to the driver via -d, and the block-
                        size env var suffix (default: fp32)
  -s, --layout LAYOUT   storage layout passed through to the driver as '-s
                        <layout>' (e.g. 'rrr', 'ccc') (default: None)
  -p, --problem-size PROBLEM_SIZE
                        problem size sweep passed to the driver as '-p
                        <value>' (default: 4000 8000 500)
  -r, --repeats REPEATS
                        repetitions per problem size, passed as '-r <value>'
                        (default: 3)
  --aggregate {max,mean,min}
                        how the GFLOPs of the problem-size sweep are combined
                        into the trial score; prefer 'mean' when tuning max
                        block sizes (--*-headroom), which only change edge
                        blocks (default: max)
  -n, --n-trials N_TRIALS
                        number of Optuna trials (default: 150)
  --seed SEED           seed for Optuna's sampler (reproducible search)
                        (default: None)
  --timeout TIMEOUT     per-benchmark wall-clock limit in seconds (default:
                        None)
  -c, --cpu-mask CPU_MASK
                        CPU mask for taskset, e.g. '0-35' or '0,2,4-7'. If
                        set, the driver is launched via 'taskset -c <mask>'.
                        Python/Optuna are not constrained. (default: None)
  --omp-places OMP_PLACES
                        OMP_PLACES value (default: '{0}:<threads>:1')
                        (default: None)
  --omp-proc-bind OMP_PROC_BIND
                        OMP_PROC_BIND value (default: true)
  -e, --env KEY=VALUE   extra environment variable for the benchmark
                        (repeatable) (default: [])
  -v, --verbose         forward the benchmark driver's stdout/stderr to the
                        terminal (useful for OMP_DISPLAY_ENV,
                        OMP_DISPLAY_AFFINITY, etc.) (default: False)
  --save-best SAVE_BEST
                        write the winning configuration to this file as shell
                        exports (default: None)
  --quiet               silence Optuna's per-trial log lines (default: False)

microkernel search space:
  The (MR, NR) register blocksize pair selects the gemm microkernel, and
  only the pairs the build contains are usable; a pair with no kernel falls
  back to the default microkernel instead of reporting an error.

  --mr-nr MRxNR         register blocksize pair (one microkernel) to try;
                        repeatable and/or comma-separated, e.g. '--mr-nr
                        8x12,12x8,16x6'. All pairs are searched as a single
                        categorical, so MR and NR are always set together and
                        mixes such as 14x12 are never tried. Sets
                        BLIS_MR_<S|D> and BLIS_NR_<S|D>; omit to leave the
                        microkernel at the build's default. (default: [])

thread decomposition search space:
  Filter the JC/IC/JR/IR decompositions before they enter the search space,
  so Optuna never spends trials on a shape that is known to be bad (0 = no
  limit).

  --jc-nt-max N         drop decompositions with BLIS_JC_NT > N (0 = keep
                        every JC_NT) (default: 0)
  --ic-nt-max N         drop decompositions with BLIS_IC_NT > N (0 = keep
                        every IC_NT) (default: 0)
  --jr-nt-max N         drop decompositions with BLIS_JR_NT > N (0 = keep
                        every JR_NT) (default: 0)
  --ir-nt-max N         drop decompositions with BLIS_IR_NT > N (0 = keep
                        every IR_NT) (default: 0)

block size search space:
  Per-parameter search bounds. --*-min is where the search range starts,
  --*-max where it ends, --*-step the granularity. --*-headroom > 0 also
  tunes the max (edge-block) value as default + 0..headroom in --*-step
  increments; 0 leaves max == default.

  --kc-min KC_MIN       lowest BLIS_KC_<S|D> value to try (default: 256)
  --kc-max KC_MAX       highest BLIS_KC_<S|D> value to try (default: 3072)
  --kc-step KC_STEP     BLIS_KC_<S|D> increment (default: 32)
  --kc-headroom KC_HEADROOM
                        how far BLIS_KC_MAX_<S|D> may exceed BLIS_KC_<S|D> (0
                        = max not tuned) (default: 0)
  --mr-in-mc-min MR_IN_MC_MIN
                        lowest BLIS_MR_IN_MC_<S|D> value to try (default: 16)
  --mr-in-mc-max MR_IN_MC_MAX
                        highest BLIS_MR_IN_MC_<S|D> value to try (default:
                        128)
  --mr-in-mc-step MR_IN_MC_STEP
                        BLIS_MR_IN_MC_<S|D> increment (default: 4)
  --mr-in-mc-headroom MR_IN_MC_HEADROOM
                        how far BLIS_MR_IN_MC_MAX_<S|D> may exceed
                        BLIS_MR_IN_MC_<S|D> (0 = max not tuned) (default: 0)
  --nr-in-nc-min NR_IN_NC_MIN
                        lowest BLIS_NR_IN_NC_<S|D> value to try (default: 128)
  --nr-in-nc-max NR_IN_NC_MAX
                        highest BLIS_NR_IN_NC_<S|D> value to try (default:
                        2048)
  --nr-in-nc-step NR_IN_NC_STEP
                        BLIS_NR_IN_NC_<S|D> increment (default: 32)
  --nr-in-nc-headroom NR_IN_NC_HEADROOM
                        how far BLIS_NR_IN_NC_MAX_<S|D> may exceed
                        BLIS_NR_IN_NC_<S|D> (0 = max not tuned) (default: 0)
```



Example running blistune:
-------------------------

```
python ./blistune.py -t 8 -o gemm -P fp64 -p "1000 2000 500" -r 4 -n 300 --omp-places "{0}:8:1" --omp-proc-bind=false --save-best avx512_3vx8.sh /data/tmp/blis-avx512-3vx8-with-knl-packm --kc-min 32 --kc-max 256 --kc-step 4 --mr-in-mc-min 40 --mr-in-mc-max 120 --mr-in-mc-step 2
driver      : /data/tmp/blis-avx512-3vx8-with-knl-packm/test/3/test_gemm_blis_mt.x
command     : ./test_gemm_blis_mt.x -d d -p '1000 2000 500' -r 4
environment : OMP_PLACES={0}:8:1  OMP_PROC_BIND=false
threads     : 8 -> 20 valid JC/IC/JR/IR decompositions
BLIS_KC_D           : 32..256 step 4
BLIS_MR_IN_MC_D     : 40..120 step 2
BLIS_NR_IN_NC_D     : 128..2048 step 32
trials      : 300

[I 2026-09-11 14:00:34,601] A new study created in memory with name: no-name-5f790061-3987-4de4-974d-cbfa45372ac1
[I 2026-09-11 14:00:35,159] Trial 0 finished with value: 449.29 and parameters: {'thread_config': '1_4_2_1', 'BLIS_KC_D': 104, 'BLIS_MR_IN_MC_D': 66, 'BLIS_NR_IN_NC_D': 800}. Best is trial 0 with value: 449.29.
[I 2026-09-11 14:00:35,691] Trial 1 finished with value: 508.09 and parameters: {'thread_config': '2_2_2_1', 'BLIS_KC_D': 212, 'BLIS_MR_IN_MC_D': 66, 'BLIS_NR_IN_NC_D': 224}. Best is trial 1 with value: 508.09.
[...]
[I 2026-09-11 14:03:15,823] Trial 298 finished with value: 509.36 and parameters: {'thread_config': '2_2_2_1', 'BLIS_KC_D': 208, 'BLIS_MR_IN_MC_D': 44, 'BLIS_NR_IN_NC_D': 1312}. Best is trial 202 with value: 533.02.
[I 2026-09-11 14:03:16,362] Trial 299 finished with value: 518.96 and parameters: {'thread_config': '2_2_2_1', 'BLIS_KC_D': 224, 'BLIS_MR_IN_MC_D': 50, 'BLIS_NR_IN_NC_D': 736}. Best is trial 202 with value: 533.02.

==================================================
BEST GFLOPs: 533.02  (trial 202)
BEST CONFIGURATION:
  BLIS_JC_NT=2
  BLIS_IC_NT=2
  BLIS_JR_NT=2
  BLIS_IR_NT=1
  BLIS_KC_D=256
  BLIS_MR_IN_MC_D=50
  BLIS_NR_IN_NC_D=1408
==================================================

wrote avx512_3vx8.sh
```

Restricting the search space
----------------------------

### Thread decompositions

Splitting IR (or any other loop) that a problem is too small for only adds
per-thread overhead. `--{jc,ic,jr,ir}-nt-max` drop decompositions *before* they
enter the search space, so Optuna never spends a trial on them (this is a
filtered categorical, not a pruned trial):

```
--ir-nt-max 1        # IR_NT is always 1; JC/IC/JR stay free
--ir-nt-max 1 --jr-nt-max 1 --jc-nt-max 4
```

0 (the default) means "no limit", which is the historical behaviour. A limit
that leaves no decomposition at all is an error, and the message lists what the
limit excluded. `-v` prints the decompositions that made it into the space.

### Microkernels (MR x NR)

`--mr-nr MRxNR` takes the register blocksize pairs to try. This aims at BLIS 
versions generated by [ukrgen](https://github.com/FZJ-JSC/jukefa-ukrgen).
The pair selects the gemm microkernel, so it is searched as a **single categorical**:
MR and NR are always set together and combinations that have no kernel (e.g. 
14x12 when only 8x12/12x8/14x8/8x14/16x6/6x16 exist) can never be generated.

```
--mr-nr 8x12,12x8,14x8,8x14,16x6,6x16        # or repeat the flag
--mr-nr "8x12 12x8 16x6"
```

Each trial sets `BLIS_MR_<S|D>` and `BLIS_NR_<S|D>` for the `--precision`
datatype. Leaving `--mr-nr` off keeps the build's default microkernel.

Caveats worth knowing:

* The pairs are **element counts**, which for VLA ISAs like RVV depend
  on the runtime vector length: a `14x2v` kernel is 14 x 2*VL elements, i.e.
  14x8 for fp32 at VLEN=256 and 14x16 at VLEN=512. Read the element counts off
  the kernel table in `config/<config>/bli_cntx_init_<config>.c`
  (`mr_vectors*vl + mr_elements` x `nr_vectors*vl + nr_elements`).
* A pair the library doesn't have does **not** fail: `ukrgen_*gemm_choose()`
  silently falls back to the default kernel, and the trial then measures the
  default microkernel instead of the requested one. Only list pairs the build
  really contains.
* blistune checks the driver binary for `BLIS_MR_<S|D>`/`BLIS_NR_<S|D>` and
  refuses to start with `--mr-nr` when the library doesn't read them.
* `BLIS_MR_IN_MC_<S|D>` and `BLIS_NR_IN_NC_<S|D>` are multipliers of MR and NR
  (`MC = MR * value`), so tuning them alongside `--mr-nr` keeps its meaning
  (it moves MC/NC with the chosen kernel, in MR/NR units).

Example:

```
python ./blistune.py -t 8 -o gemm -P fp32 -n 400 --aggregate mean \
    --ir-nt-max 1 \
    --mr-nr 8x12,12x8,14x8,8x14,16x6,6x16 \
    --mr-in-mc-min 4 --mr-in-mc-max 32 --mr-in-mc-step 2 \
    --save-best rvv_x100_gemm_fp32.sh /path/to/compiled/blis
```

whose header reports the filtered spaces:

```
threads     : 8 -> 10 valid JC/IC/JR/IR decompositions  [IR_NT<=1 applied, 10 excluded]
microkernel : 6 BLIS_MR_S/BLIS_NR_S pairs -> 8x12 12x8 14x8 8x14 16x6 6x16
```


# License

blistune is distributed under the terms of both the MIT license and the GNU General Public License v3.0. Users may choose either license, at their option.

All new contributions must be made under both the MIT and GNU General Public License v3.0.

See LICENSE-GPL-3.0, LICENSE-MIT for details.

SPDX-License-Identifier: MIT OR GPL-3.0-or-later
