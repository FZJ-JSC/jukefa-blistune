#!/usr/bin/env python3
"""
BLIS block size / threading autotuner.

Drives one of the BLIS multithreaded test drivers
(<blis_build_dir>/test/3/test_<op>_blis_mt.x) under Optuna, searching over

  * the 4-way thread decomposition (BLIS_JC_NT / BLIS_IC_NT / BLIS_JR_NT / BLIS_IR_NT),
    optionally restricted per loop (e.g. --ir-nt-max 1)
  * the microkernel, as a BLIS_MR_{S,D} x BLIS_NR_{S,D} pair taken from an
    explicit list of pairs the build actually contains
  * the cache block sizes (BLIS_KC_{S,D}, BLIS_MR_IN_MC_{S,D}, BLIS_NR_IN_NC_{S,D})
  * optionally, their max (edge-block) values (BLIS_KC_MAX_{S,D}, ...), searched
    as a headroom above the default so that max >= default always holds

and maximising reported GFLOPs.

Requires a BLIS built with runtime-configurable block sizes; the script checks
the driver binary for the relevant environment variable names (or the format
strings the bli_gks.c override code builds them from) before starting.
"""

import argparse
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

try:
    import optuna
except ImportError:  # pragma: no cover
    sys.exit("optuna is required: pip install optuna")


# --------------------------------------------------------------------------- #
# Static tables
# --------------------------------------------------------------------------- #

OPERATIONS = ("gemm", "hemm", "herk", "trmm", "trsm")

# precision -> (driver -d char, BLIS env var suffix, element size in bytes)
PRECISIONS = {
    "fp32": ("s", "S", 4),
    "fp64": ("d", "D", 8),
}

# Thread loops, in the order used by the "jc_ic_jr_ir" categorical encoding.
THREAD_LOOPS = ("JC", "IC", "JR", "IR")

# cli stem -> (BLIS env var stem, default min, default max, default step)
BLOCK_PARAMS = {
    "kc": ("KC", 256, 3072, 32),
    "mr-in-mc": ("MR_IN_MC", 16, 128, 4),
    "nr-in-nc": ("NR_IN_NC", 128, 2048, 32),
}

# --tile search space: the A and B panel sizes in bytes -> (min, max, step).
# A panel = MC*KC*esize, B panel = NC*KC*esize. Those products are the cache
# capacity invariants, so their optimum does not move with the microkernel the
# way a multiple of MR/NR does.
TILE_PARAMS = {
    "a-tile": (64 << 10, 1 << 20, 32 << 10),
    "b-tile": (256 << 10, 8 << 20, 256 << 10),
}

# --tile stem -> (block size stem whose env var it derives, MR or NR as the unit)
TILE_TARGETS = {
    "a-tile": ("mr-in-mc", "MR"),
    "b-tile": ("nr-in-nc", "NR"),
}

# Matches a BLIS test-driver result row, e.g.
#   data_gemm_blis( 3, 1:4 ) = [ 4000 4000 4000  123.45 ];
# capturing the dimension columns and the trailing GFLOPs value. Tolerates 1-4
# leading dimension columns (herk/trmm/trsm print fewer). The drivers also print
# an all-zero placeholder row before the results; the parser skips it.
RESULT_RE = re.compile(r"\[\s*((?:\d+\s+){1,4})([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*\]")

# How the GFLOPs of the problem-size sweep are combined into one trial score.
AGGREGATES = {
    "max": max,
    "mean": statistics.fmean,
    "min": min,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def thread_configs(num_threads):
    """All (jc, ic, jr, ir) tuples whose product is exactly num_threads."""
    configs = []
    for jc in divisors(num_threads):
        rem_jc = num_threads // jc
        for ic in divisors(rem_jc):
            rem_ic = rem_jc // ic
            for jr in divisors(rem_ic):
                ir = rem_ic // jr
                configs.append((jc, ic, jr, ir))
    return configs


def restrict_thread_configs(configs, limits):
    """Drop the decompositions whose ways exceed a per-loop maximum.

    `limits` maps a loop name (JC, IC, JR, IR) to its maximum allowed way count;
    a limit of 0 (or None) means "unlimited". Splitting a loop that the matrix
    is too small for only adds per-thread overhead, so e.g. limits={'IR': 1}
    keeps IR_NT == 1 everywhere while leaving the other three loops free.
    """
    active = {loop: lim for loop, lim in limits.items() if lim}
    if not active:
        return list(configs)

    kept = []
    for cfg in configs:
        values = dict(zip(THREAD_LOOPS, cfg))
        if all(values[loop] <= lim for loop, lim in active.items()):
            kept.append(cfg)
    return kept


def parse_mr_nr(specs):
    """Parse --mr-nr values ('8x12,12x8' / repeated) into ordered (mr, nr) pairs.

    The pair is what identifies a microkernel: not every (MR, NR) combination
    has one, and BLIS / ukrgen silently falls back to the default kernel for a
    pair it doesn't know, so only pairs the caller lists are ever searched.
    """
    pairs = []
    for spec in specs:
        for token in re.split(r"[,\s]+", spec.replace("X", "x")):
            token = token.strip()
            if not token:
                continue
            match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", token)
            if not match:
                raise SystemExit(f"--mr-nr expects MRxNR (e.g. 16x6), got {token!r}")
            pair = (int(match.group(1)), int(match.group(2)))
            if pair[0] < 1 or pair[1] < 1:
                raise SystemExit(f"--mr-nr needs positive MR and NR, got {token!r}")
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def tile_to_multiplier(panel_bytes, esize, unit, kc):
    """BLIS_MR_IN_MC/NR_IN_NC that makes the panel `panel_bytes` bytes wide.

    A panel is `unit * multiplier` rows of `kc` elements, one element being
    `esize` bytes; MC = MR * multiplier, so this is the inverse of that. BLIS
    requires the multiplier to be at least 1, hence the clamp.
    """
    return max(1, round(panel_bytes / (esize * unit * kc)))


def binary_contains(path, needles, chunk_size=1 << 20):
    """Return the subset of `needles` (bytes) present in the file at `path`."""
    needles = list(needles)
    overlap = max(len(n) for n in needles) - 1
    found = set()
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            buf = tail + chunk
            for needle in needles:
                if needle not in found and needle in buf:
                    found.add(needle)
            if len(found) == len(needles):
                break
            tail = buf[-overlap:] if overlap else b""
    return found


def align_range(low, high, step, label):
    """Clamp `high` down to low + k*step so Optuna doesn't silently adjust it."""
    if high < low:
        raise SystemExit(f"--{label}-max ({high}) must be >= --{label}-min ({low})")
    if step <= 0:
        raise SystemExit(f"--{label}-step must be positive")
    span = ((high - low) // step) * step
    return low, low + span


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="Autotune BLIS block sizes and thread decomposition with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("blis_build_dir", type=Path,
                   help="BLIS build directory (drivers are looked up under <dir>/test/3)")

    p.add_argument("-t", "--threads", type=int, required=True,
                   help="total thread count; the JC/IC/JR/IR product must equal this")
    p.add_argument("-o", "--operation", choices=OPERATIONS, default="gemm",
                   help="which test_<op>_blis_mt.x driver to benchmark")
    p.add_argument("-P", "--precision", choices=sorted(PRECISIONS), default="fp32",
                   help="datatype passed to the driver via -d, and the block-size env var suffix")
    p.add_argument("-s", "--layout", default=None,
                   help="storage layout passed through to the driver as '-s <layout>' (e.g. 'rrr', 'ccc')")

    p.add_argument("-p", "--problem-size", default="4000 8000 500",
                   help="problem size sweep passed to the driver as '-p <value>'")
    p.add_argument("-r", "--repeats", type=int, default=3,
                   help="repetitions per problem size, passed as '-r <value>'")
    p.add_argument("--aggregate", choices=sorted(AGGREGATES), default="max",
                   help="how the GFLOPs of the problem-size sweep are combined into the "
                        "trial score; prefer 'mean' when tuning max block sizes "
                        "(--*-headroom), which only change edge blocks")

    p.add_argument("-n", "--n-trials", type=int, default=150,
                   help="number of Optuna trials")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for Optuna's sampler (reproducible search)")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-benchmark wall-clock limit in seconds")

    p.add_argument("-c", "--cpu-mask", default=None,
                   help="CPU mask for taskset, e.g. '0-35' or '0,2,4-7'. "
                        "If set, the driver is launched via "
                        "'taskset -c <mask>'. Python/Optuna are not constrained.")

    p.add_argument("--omp-places", default=None,
                   help="OMP_PLACES value (default: '{0}:<threads>:1')")
    p.add_argument("--omp-proc-bind", default="true",
                   help="OMP_PROC_BIND value")
    p.add_argument("-e", "--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra environment variable for the benchmark (repeatable)")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="forward the benchmark driver's stdout/stderr to the terminal "
                        "(useful for OMP_DISPLAY_ENV, OMP_DISPLAY_AFFINITY, etc.)")

    p.add_argument("--save-best", type=Path, default=None,
                   help="write the winning configuration to this file as shell exports")
    p.add_argument("--quiet", action="store_true",
                   help="silence Optuna's per-trial log lines")

    mode = p.add_argument_group(
        "search strategy",
        "Both modes need --mr-nr, since MR/NR identify the microkernel and are "
        "fixed by the ukrgen configuration rather than by BLIS itself.",
    )
    mode.add_argument("--tile", action="store_true",
                      help="search the A/B panel sizes in bytes and derive "
                           "BLIS_MR_IN_MC_<S|D>/BLIS_NR_IN_NC_<S|D> from them "
                           "(MC*KC and NC*KC are the cache invariants, so the "
                           "optimum no longer moves with MR/NR). "
                           "--mr-in-mc-*/--nr-in-nc-* are unused in this mode.")
    mode.add_argument("--per-kernel", action="store_true",
                      help="run one study per --mr-nr pair, splitting --n-trials "
                           "between them, and report every pair's own best "
                           "configuration. Prevents TPE from concentrating the "
                           "whole budget on the microkernel that looked good early.")

    micro = p.add_argument_group(
        "microkernel search space",
        "The (MR, NR) register blocksize pair selects the gemm microkernel, and "
        "only the pairs the build contains are usable; a pair with no kernel "
        "falls back to the default microkernel instead of reporting an error.",
    )
    micro.add_argument("--mr-nr", action="append", default=[], metavar="MRxNR",
                       help="register blocksize pair (one microkernel) to try; "
                            "repeatable and/or comma-separated, e.g. "
                            "'--mr-nr 8x12,12x8,16x6'. All pairs are searched as a "
                            "single categorical, so MR and NR are always set "
                            "together and mixes such as 14x12 are never tried. "
                            "Sets BLIS_MR_<S|D> and BLIS_NR_<S|D>; omit to leave "
                            "the microkernel at the build's default.")

    decomp = p.add_argument_group(
        "thread decomposition search space",
        "Filter the JC/IC/JR/IR decompositions before they enter the search "
        "space, so Optuna never spends trials on a shape that is known to be "
        "bad (0 = no limit).",
    )
    for loop in THREAD_LOOPS:
        decomp.add_argument(f"--{loop.lower()}-nt-max", type=int, default=0,
                            metavar="N",
                            help=f"drop decompositions with BLIS_{loop}_NT > N "
                                 f"(0 = keep every {loop}_NT)")

    tiles = p.add_argument_group(
        "tile size search space (--tile)",
        "Panel sizes in bytes: the A panel is MC*KC*<element size>, the B panel "
        "NC*KC*<element size>. MC/NC and the BLIS_MR_IN_MC_/BLIS_NR_IN_NC_ "
        "multipliers are derived from them per trial.",
    )
    for stem, (dmin, dmax, dstep) in TILE_PARAMS.items():
        label = f"the {stem[0].upper()} panel size in bytes"
        tiles.add_argument(f"--{stem}-min", type=int, default=dmin,
                           help=f"smallest {label}")
        tiles.add_argument(f"--{stem}-max", type=int, default=dmax,
                           help=f"largest {label}")
        tiles.add_argument(f"--{stem}-step", type=int, default=dstep,
                           help=f"{stem[0].upper()} panel size increment in bytes")

    group = p.add_argument_group(
        "block size search space",
        "Per-parameter search bounds. --*-min is where the search range starts, "
        "--*-max where it ends, --*-step the granularity. --*-headroom > 0 also "
        "tunes the max (edge-block) value as default + 0..headroom in --*-step "
        "increments; 0 leaves max == default.",
    )
    for stem, (blis_stem, dmin, dmax, dstep) in BLOCK_PARAMS.items():
        var = f"BLIS_{blis_stem}_<S|D>"
        group.add_argument(f"--{stem}-min", type=int, default=dmin,
                           help=f"lowest {var} value to try")
        group.add_argument(f"--{stem}-max", type=int, default=dmax,
                           help=f"highest {var} value to try")
        group.add_argument(f"--{stem}-step", type=int, default=dstep,
                           help=f"{var} increment")
        group.add_argument(f"--{stem}-headroom", type=int, default=0,
                           help=f"how far BLIS_{blis_stem}_MAX_<S|D> may exceed {var} "
                                f"(0 = max not tuned)")

    return p


def parse_extra_env(pairs):
    env = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #

class Benchmark:
    def __init__(self, exe, workdir, cmd, base_env, timeout, verbose=False,
                 aggregate="max"):
        self.exe = exe
        self.workdir = workdir
        self.cmd = cmd
        self.base_env = base_env
        self.timeout = timeout
        self.verbose = verbose
        self.aggregate = AGGREGATES[aggregate]

    def run(self, config):
        env = os.environ.copy()
        env.update(self.base_env)
        env.update({k: str(v) for k, v in config.items()})

        try:
            result = subprocess.run(
                self.cmd,
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"  ! timed out after {self.timeout}s", file=sys.stderr)
            if self.verbose:
                if exc.stdout:
                    sys.stdout.write(exc.stdout)
                    sys.stdout.flush()
                if exc.stderr:
                    sys.stderr.write(exc.stderr)
                    sys.stderr.flush()
            return 0.0
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or exc.stdout or "").strip().splitlines()[-3:]
            print(f"  ! driver exited {exc.returncode}: {' / '.join(tail)}", file=sys.stderr)
            if self.verbose:
                if exc.stdout:
                    sys.stdout.write(exc.stdout)
                    sys.stdout.flush()
                if exc.stderr:
                    sys.stderr.write(exc.stderr)
                    sys.stderr.flush()
            return 0.0

        # ---- verbose forwarding ------------------------------------------ #
        # The driver's full stdout/stderr is normally swallowed here. When
        # --verbose is set, write it through so OpenMP diagnostics
        # (OMP_DISPLAY_ENV, OMP_DISPLAY_AFFINITY, OMP_AFFINITY_FORMAT) and the
        # driver's own tables show up in the terminal.
        if self.verbose:
            if result.stdout:
                sys.stdout.write(result.stdout)
                sys.stdout.flush()
            if result.stderr:
                sys.stderr.write(result.stderr)
                sys.stderr.flush()

        # ---- parse GFLOP/s from stdout (always) --------------------------- #
        # Rows whose dimensions are all zero are the drivers' placeholders;
        # they'd drag 'mean' and 'min' down.
        values = [float(gflops) for dims, gflops in RESULT_RE.findall(result.stdout)
                  if any(int(d) for d in dims.split())]
        if not values:
            print("  ! no parseable result rows in driver output", file=sys.stderr)
            return 0.0
        return self.aggregate(values)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")

    dtype_char, suffix, esize = PRECISIONS[args.precision]

    # ---- search space: microkernel (MR x NR) ------------------------------ #
    # MR and NR are a single choice: the pair identifies the microkernel, so
    # they must never be explored independently.
    mr_nr_pairs = parse_mr_nr(args.mr_nr)
    mr_nr_env = (f"BLIS_MR_{suffix}", f"BLIS_NR_{suffix}")
    mr_nr_labels = [f"{mr}x{nr}" for mr, nr in mr_nr_pairs]

    # Both search strategies below convert between MR/NR and the blocksizes, so
    # they need the microkernel to be known rather than left to the build.
    if args.tile and not mr_nr_pairs:
        raise SystemExit(
            "--tile needs the microkernel: a panel size is turned into "
            "BLIS_MR_IN_MC_<S|D>/BLIS_NR_IN_NC_<S|D> via MR/NR. Pass --mr-nr."
        )
    if args.per_kernel and not mr_nr_pairs:
        raise SystemExit(
            "--per-kernel stages the search over the microkernels, so pass --mr-nr."
        )
    if args.per_kernel and args.n_trials < len(mr_nr_pairs):
        raise SystemExit(
            f"--per-kernel splits --n-trials ({args.n_trials}) over "
            f"{len(mr_nr_pairs)} microkernels, i.e. less than one trial each; "
            f"raise --n-trials or drop --per-kernel."
        )

    # ---- locate the driver ------------------------------------------------ #
    workdir = (args.blis_build_dir / "test" / "3").resolve()
    exe_name = f"test_{args.operation}_blis_mt.x"
    exe = workdir / exe_name
    if not exe.is_file():
        raise SystemExit(
            f"benchmark not found: {exe}\n"
            f"(is {args.blis_build_dir} a BLIS build dir, and were the test drivers built?)"
        )
    if not os.access(exe, os.X_OK):
        raise SystemExit(f"benchmark is not executable: {exe}")

    # ---- confirm the binary understands the env vars we intend to set ----- #
    # Per-config overrides leave the literal names (e.g. "BLIS_KC_D") in the
    # binary. The generic override in bli_gks.c builds them at run time from
    # "BLIS_%s_%s" / "BLIS_%s_MAX_%s", so look for those format strings too;
    # unlike a static function's name, they survive inlining and stripping.
    block_env_names = {
        stem: f"BLIS_{blis_stem}_{suffix}"
        for stem, (blis_stem, _, _, _) in BLOCK_PARAMS.items()
    }
    max_env_names = {
        stem: f"BLIS_{blis_stem}_MAX_{suffix}"
        for stem, (blis_stem, _, _, _) in BLOCK_PARAMS.items()
    }
    needles = [name.encode() for name in block_env_names.values()]
    max_needles = [name.encode() for name in max_env_names.values()]
    fmt_needles = [b"BLIS_%s_%s", b"MR_IN_MC"]
    fmt_max_needle = b"BLIS_%s_MAX_%s"
    mr_nr_needles = [name.encode() for name in mr_nr_env] if mr_nr_pairs else []
    present = binary_contains(
        exe, needles + max_needles + fmt_needles + mr_nr_needles + [fmt_max_needle]
    )
    missing = sorted(n.decode() for n in needles if n not in present)
    has_gks_env = all(n in present for n in fmt_needles)
    has_max_env = fmt_max_needle in present or all(n in present for n in max_needles)
    if missing and not has_gks_env:
        raise SystemExit(
            f"this blis library doesn't support dynamic block sizes\n"
            f"  binary:  {exe}\n"
            f"  missing: {', '.join(missing)}\n"
            f"Rebuild BLIS with runtime-configurable block sizes, or point "
            f"--precision at the datatype this build was configured for."
        )

    # MR/NR are read by the sub-configuration (ukrgen's bli_cntx_init_*.c), not
    # by the generic bli_gks.c override, so the names may legitimately be absent
    # from a build that only supports dynamic cache blocksizes.
    mr_nr_missing = sorted(n.decode() for n in mr_nr_needles if n not in present)
    if mr_nr_missing:
        raise SystemExit(
            f"this blis library doesn't allow selecting the microkernel at run time\n"
            f"  binary:  {exe}\n"
            f"  missing: {', '.join(mr_nr_missing)}\n"
            f"Rebuild BLIS from a ukrgen configuration that reads BLIS_MR_/BLIS_NR_ "
            f"for the {suffix} datatype, or drop --mr-nr."
        )

    # ---- search space ----------------------------------------------------- #
    limits = {loop: getattr(args, f"{loop.lower()}_nt_max") for loop in THREAD_LOOPS}
    for loop, limit in limits.items():
        if limit < 0:
            raise SystemExit(f"--{loop.lower()}-nt-max must be >= 0")

    all_thread_space = thread_configs(args.threads)
    if not all_thread_space:
        raise SystemExit(f"no valid 4-way thread decomposition for {args.threads} threads")
    thread_space = restrict_thread_configs(all_thread_space, limits)
    if not thread_space:
        raise SystemExit(
            f"--*-nt-max excludes all {len(all_thread_space)} decompositions of "
            f"{args.threads} threads:\n  "
            + "\n  ".join("_".join(str(v) for v in cfg) for cfg in all_thread_space)
        )
    thread_choices = ["_".join(str(v) for v in cfg) for cfg in thread_space]
    limit_desc = ", ".join(f"{loop}_NT<={limit}" for loop, limit in limits.items() if limit)

    block_space = {}
    for stem in BLOCK_PARAMS:
        if args.tile and stem != "kc":
            # MC and NC come from the panel sizes in tile mode; their own
            # multiplier bounds would only fight the derivation.
            continue
        attr = stem.replace("-", "_")
        low = getattr(args, f"{attr}_min")
        high = getattr(args, f"{attr}_max")
        step = getattr(args, f"{attr}_step")
        low, high = align_range(low, high, step, stem)
        block_space[stem] = (low, high, step)

    tile_space = {}
    if args.tile:
        for stem in TILE_PARAMS:
            attr = stem.replace("-", "_")
            low = getattr(args, f"{attr}_min")
            high = getattr(args, f"{attr}_max")
            step = getattr(args, f"{attr}_step")
            low, high = align_range(low, high, step, stem)
            tile_space[stem] = (low, high, step)

    # Largest KC that will be tried: the panel -> multiplier resolution is
    # esize*unit*KC bytes, so this is where a byte grid is coarsest.
    kc_high = block_space["kc"][1]

    # Max (edge-block) values are searched as a headroom above the default, in
    # the default's step. That keeps each parameter's range fixed across
    # trials, and max >= default holds by construction (the patched BLIS
    # aborts at init otherwise). Headroom 0 is the old max == default.
    max_space = {}
    for stem in BLOCK_PARAMS:
        headroom = getattr(args, f"{stem.replace('-', '_')}_headroom")
        if stem not in block_space:
            # tile mode: the multiplier is derived, so it has no step for the
            # headroom arithmetic; edge blocks are then not tuned for it.
            if headroom:
                raise SystemExit(
                    f"--{stem}-headroom is not available with --tile, where "
                    f"BLIS_{BLOCK_PARAMS[stem][0]}_{suffix} is derived from the "
                    f"panel size; drop it or leave --tile off."
                )
            continue
        step = block_space[stem][2]
        if headroom < 0:
            raise SystemExit(f"--{stem}-headroom must be >= 0")
        if headroom == 0:
            continue
        if headroom < step:
            raise SystemExit(f"--{stem}-headroom ({headroom}) must be 0 or at least "
                             f"--{stem}-step ({step})")
        max_space[stem] = (0, (headroom // step) * step, step)

    if max_space and not has_max_env:
        raise SystemExit(
            f"this blis library doesn't read the max block size variables\n"
            f"  binary:  {exe}\n"
            f"  wanted:  {', '.join(max_env_names[stem] for stem in max_space)}\n"
            f"Rebuild BLIS with the MAX-aware bli_gks.c patch, or drop --*-headroom."
        )

    # ---- command + environment -------------------------------------------- #
    cmd = [f"./{exe_name}", "-d", dtype_char, "-p", args.problem_size, "-r", str(args.repeats)]
    if args.layout:
        cmd += ["-s", args.layout]

    # Wrap the driver in taskset if a CPU mask was given. Only the driver is
    # constrained; Python and Optuna keep their normal scheduling freedom.
    if args.cpu_mask:
        cmd = ["taskset", "-c", args.cpu_mask] + cmd

    base_env = {
        "OMP_PLACES": args.omp_places or f"{{0}}:{args.threads}:1",
        "OMP_PROC_BIND": args.omp_proc_bind,
    }
    base_env.update(parse_extra_env(args.env))

    benchmark = Benchmark(exe, workdir, cmd, base_env, args.timeout,
                          verbose=args.verbose, aggregate=args.aggregate)

    # ---- objective --------------------------------------------------------- #
    def build_config(trial, fixed_pair=None):
        """Suggest one trial's parameters; return its {BLIS variable: value}.

        `fixed_pair` pins the microkernel, which is what --per-kernel does: the
        regime stops being a search dimension and becomes the study itself.
        """
        # The microkernel comes first: (MR, NR) is one categorical so the two
        # variables move together and a pair without a kernel is impossible.
        config = {}
        if fixed_pair is not None:
            config[mr_nr_env[0]], config[mr_nr_env[1]] = fixed_pair
        elif mr_nr_pairs:
            pair = trial.suggest_categorical("mr_nr", mr_nr_labels)
            config[mr_nr_env[0]], config[mr_nr_env[1]] = map(int, pair.split("x"))

        thread_str = trial.suggest_categorical("thread_config", thread_choices)
        ways = dict(zip(THREAD_LOOPS, map(int, thread_str.split("_"))))
        config.update({f"BLIS_{loop}_NT": ways[loop] for loop in THREAD_LOOPS})

        for stem, (low, high, step) in block_space.items():
            env_name = block_env_names[stem]
            config[env_name] = trial.suggest_int(env_name, low, high, step=step)

        # --tile: the panel size is the search dimension, MC/NC follow from it
        # (MC = MR * MR_IN_MC and the A panel is MC*KC*esize bytes, so the
        # multiplier is just the panel measured in MR-sized rows).
        if tile_space:
            mr = config[mr_nr_env[0]]
            nr = config[mr_nr_env[1]]
            kc = config[block_env_names["kc"]]
            for stem, (low, high, step) in tile_space.items():
                target_stem, unit_name = TILE_TARGETS[stem]
                unit = mr if unit_name == "MR" else nr
                panel = trial.suggest_int(stem.replace("-", "_"), low, high, step=step)
                config[block_env_names[target_stem]] = tile_to_multiplier(
                    panel, esize, unit, kc
                )

        for stem, (low, high, step) in max_space.items():
            headroom = trial.suggest_int(f"{stem}-headroom", low, high, step=step)
            config[max_env_names[stem]] = config[block_env_names[stem]] + headroom

        # The max values (and, with --tile, MC/NC) are derived rather than
        # Optuna parameters, so keep the exact environment of every trial.
        trial.set_user_attr("config", config)
        return config

    def run_trial(trial, fixed_pair=None):
        config = build_config(trial, fixed_pair)

        if args.verbose:
            print(f"\n[trial {trial.number}] config: "
                  + " ".join(f"{k}={v}" for k, v in sorted(config.items())),
                  file=sys.stderr)

        return benchmark.run(config)

    # ---- run --------------------------------------------------------------- #
    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"driver      : {exe}")
    print(f"command     : {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"environment : " + "  ".join(f"{k}={v}" for k, v in base_env.items()))
    if args.cpu_mask:
        print(f"taskset     : -c {args.cpu_mask}")
    print(f"threads     : {args.threads} -> {len(thread_space)} valid JC/IC/JR/IR decompositions"
          + (f"  [{limit_desc} applied, {len(all_thread_space) - len(thread_space)} excluded]"
             if limit_desc else ""))
    if mr_nr_pairs:
        print(f"microkernel : {len(mr_nr_pairs)} {mr_nr_env[0]}/{mr_nr_env[1]} pairs -> "
              + " ".join(mr_nr_labels))
    else:
        print(f"microkernel : build default (BLIS_MR_{suffix}/BLIS_NR_{suffix} not set)")
    if args.tile:
        print("mode        : tile panels, MC/NC derived per trial "
              "(--mr-in-mc-*/--nr-in-nc-* unused)")
    if args.per_kernel:
        print(f"mode        : one study per microkernel, {len(mr_nr_pairs)} studies "
              f"over {args.n_trials} trials")
    for stem, (low, high, step) in block_space.items():
        print(f"{block_env_names[stem]:<20}: {low}..{high} step {step}")
    for stem, (low, high, step) in max_space.items():
        print(f"{max_env_names[stem]:<20}: {block_env_names[stem]} + {low}..{high} step {step}")
    for stem, (low, high, step) in tile_space.items():
        target = block_env_names[TILE_TARGETS[stem][0]]
        print(f"{stem:<20}: {low}..{high} step {step} bytes -> {target}")
        # A panel only moves in whole multipliers, i.e. in quanta of
        # esize*unit*KC bytes, so at a large KC a byte grid collapses onto a
        # handful of multipliers. Say so instead of letting it look like a
        # fine-grained search.
        unit_name = TILE_TARGETS[stem][1]
        spans = []
        for mr, nr in mr_nr_pairs:
            unit = mr if unit_name == "MR" else nr
            spans.append(f"{mr}x{nr}:{tile_to_multiplier(low, esize, unit, kc_high)}"
                         f"..{tile_to_multiplier(high, esize, unit, kc_high)}")
        narrow = any(
            tile_to_multiplier(high, esize,
                               mr if unit_name == "MR" else nr, kc_high)
            - tile_to_multiplier(low, esize,
                                 mr if unit_name == "MR" else nr, kc_high) < 3
            for mr, nr in mr_nr_pairs
        )
        print(f"{'':<20}  {target} range at KC={kc_high}: " + " ".join(spans)
              + ("   [narrow: raise --" + stem + "-max]" if narrow else ""))
    print(f"score       : {args.aggregate} GFLOPs over the problem-size sweep")
    if max_space and args.aggregate == "max":
        print("note        : max block sizes only change edge blocks, which '--aggregate max' "
              "mostly ignores; consider --aggregate mean")
    print(f"trials      : {args.n_trials}")
    if args.verbose:
        print(f"verbose     : on (forwarding driver stdout/stderr)")
        print(f"decomps     : " + " ".join(thread_choices))
    print()

    studies = []
    if args.per_kernel:
        # Split the budget as evenly as possible; the first kernels take the
        # remainder. Each study owns one microkernel, so block sizes never have
        # to compete across regimes and every kernel gets mapped.
        shares = [args.n_trials // len(mr_nr_pairs)] * len(mr_nr_pairs)
        for i in range(args.n_trials % len(mr_nr_pairs)):
            shares[i] += 1
        for i, (pair, trials) in enumerate(zip(mr_nr_pairs, shares)):
            sampler = (None if args.seed is None
                       else optuna.samplers.TPESampler(seed=args.seed + i))
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda trial, pair=pair: run_trial(trial, pair),
                           n_trials=trials)
            studies.append((pair, study))
        study = max((s for _, s in studies), key=lambda s: s.best_value)
    else:
        sampler = (None if args.seed is None
                   else optuna.samplers.TPESampler(seed=args.seed))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(run_trial, n_trials=args.n_trials)

    # ---- report ------------------------------------------------------------ #
    # Export the environment the best trial actually ran with: thread ways,
    # block sizes and any derived max values (default + headroom).
    exports = [f"{k}={v}" for k, v in study.best_trial.user_attrs["config"].items()]

    if studies:
        # Per-kernel results are the point of --per-kernel: the winning study
        # alone would hide what the other microkernels can do.
        print("\nPER-MICROKERNEL RESULTS")
        print(f"  {'MRxNR':<8}{'GFLOPs':>9}{'trials':>8}  {'threads':<10}blocksizes")
        for pair, sub in studies:
            cfg = sub.best_trial.user_attrs["config"]
            ways = "_".join(str(cfg[f"BLIS_{loop}_NT"]) for loop in THREAD_LOOPS)
            blocks = " ".join(f"{k.replace('BLIS_', '')}={v}"
                              for k, v in sorted(cfg.items()) if "_NT" not in k)
            print(f"  {pair[0]:>2}x{pair[1]:<2} {sub.best_value:9.2f}"
                  f"{len(sub.trials):>8}  {ways:<10}{blocks}")
        winner = study.best_trial.user_attrs["config"]
        print(f"  winner: {winner[mr_nr_env[0]]}x{winner[mr_nr_env[1]]} "
              f"at {study.best_value:.2f} GFLOPs")

    print("\n" + "=" * 50)
    winner_note = ""
    if studies:
        won = study.best_trial.user_attrs["config"]
        winner_note = f", {won[mr_nr_env[0]]}x{won[mr_nr_env[1]]}"
    print(f"BEST GFLOPs: {study.best_value:.2f}  "
          f"(trial {study.best_trial.number}{winner_note})")
    print("BEST CONFIGURATION:")
    for line in exports:
        print(f"  {line}")
    print("=" * 50)

    if args.save_best:
        args.save_best.write_text(
            "\n".join(
                [
                    f"# {args.operation} / {args.precision} / {args.threads} threads",
                    f"# {study.best_value:.2f} GFLOPs",
                    f"export OMP_PLACES='{base_env['OMP_PLACES']}'",
                    f"export OMP_PROC_BIND={base_env['OMP_PROC_BIND']}",
                ]
                + [f"export {line}" for line in exports]
            )
            + "\n"
        )
        print(f"\nwrote {args.save_best}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
