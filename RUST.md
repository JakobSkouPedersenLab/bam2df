# bam2df in Rust: design and implementation plan

This document is the plan for porting `bam2df` from Python to Rust. It is written so that an
implementation session can pick it up cold: it states what to build, how the Python behaves
(including its quirks), how to prove the Rust output matches, and the steps in which to do it.

**Parity reference:** the Python code at commit `437da1c`, the last of the fixes listed in
[section 3.1](#31-bugs-fixed-in-python-before-the-port). Where this document and that code disagree,
the code wins, unless the disagreement is listed under [Decisions](#3-decisions).

---

## 0. How to use this document

Each implementation session:

1. Reads this file, then the [progress log](#10-progress-log) at the bottom to find the next step.
2. Implements that step only, with its unit tests, and inspects the diff statically. It does not run
   any Rust command on the Windows machine (see below).
3. Hands the user the step's **server test** commands from [section 8](#8-steps).
4. The user runs them on the server and pastes the output back. A step is done when its pass
   criteria hold. With the user's approval, the session commits the step and updates the progress
   log.

Rule IDs such as **R-PAIR-3** refer to the behaviour rules in [section 5](#5-behaviour-spec-the-parity-contract).
Cite them in code comments where the Rust reproduces a non-obvious Python behaviour.

### Where things run

| Machine | Has | Used for |
|---|---|---|
| Local Windows dev machine | conda env `dvfm` with polars, numpy, scipy, pyarrow, pytest; no pysam or pyBigWig (no Windows build). No Rust toolchain. | Writing code. Python tests that need no BAM run here (`tests/conftest.py` stubs pysam and pyBigWig and skips the tests marked `requires_pysam`). No Rust command is run here: no `cargo`, `rustc` or `rustfmt`. |
| Server (Linux, SLURM, gwf) | conda env `bam2df` with pysam, polars, pyarrow; Rust toolchain; real data | Everything that builds or runs: the full Python suite, golden outputs, `cargo build`, `cargo test`, `cargo fmt`. |

On the server, every command that builds, tests or reads data runs on a compute node through `srun`,
never on the frontend. The commands in this document are written that way, with `$ACCOUNT` standing
for the SLURM account. gwf targets already run on compute nodes.

Patient data never leaves the server and is never committed. Only synthetic fixtures (section 7.1)
go into the repository.

---

## 1. Goals

- A single binary `bam2df` that takes the same arguments as `src/bam2df.py` and writes the same
  `<stem>.parquet` and `<stem>_metadata.csv`.
- **Parity:** on the same inputs, the Rust output has the same schema (names, order, Arrow types)
  and the same multiset of rows as the Python output. Non-float columns are equal exactly; float
  columns are equal within the tolerance in section 7.3. Row order may differ.
- Parallelism of one job per chromosome, on 12 cores under gwf, scheduled longest chromosome first.
- Python remains in the repository and runnable until the Rust version is validated (step 7).

Not goals for the port itself: changing features, fixing the quirks in section 6 (that is step 8,
after parity), deterministic match subsampling parity with Python (D11).

---

## 2. Architecture

### 2.1 Workspace layout

```
Cargo.toml                    # [workspace] members = ["crates/*"]; commit Cargo.lock
crates/
  bam2df-core/                # library: all logic
    src/
      lib.rs
      config.rs               # BamConfig (serde), validation                    R-CFG-*
      ids.rs                  # sample/patient ID resolution                      R-ID-*
      bamio.rs                # the only module that touches noodles BAM types
      read.rs                 # ReadView: per-read decoded arrays, CIGAR/MD walks R-READ-*
      filters.rs              # read filters and type predicates                 R-FILT-*
      pairing.rs              # MateBuffer with FIFO eviction                    R-PAIR-*
      extract.rs              # indel / mismatch / match observations, n_opp     R-EXT-*
      features.rs             # observation -> output row values                 R-ROW-*
      reference.rs            # FASTA window cache                               R-REF-*
      depth.rs                # read depth matching pysam count_coverage         R-DEPTH-*
      tracks/
        bigwig.rs             # windowed bigWig value cache                      R-TRK-1..2
        annotation.rs         # microsatellite and RepeatMasker lookups          R-TRK-3..5
        chromatin.rs          # npz reader, open-bin features                    R-TRK-6..8
      inputs/
        intervals.rs          # IntervalSet; genomic/donor/patient blacklists    R-IN-1..5
        germline.rs           # variant blacklist + reference check              R-IN-6..9
        bed_regions.rs        # --bed_intervals_path                             R-IN-10
        mutations.rs          # mutation catalog TSV                             R-IN-11
      output/
        schema.rs             # Arrow schema, column order                       R-OUT-*
        builder.rs            # columnar RowBuilder -> RecordBatch
        writer.rs             # part-file writer (zstd)
        concat.rs             # row-group copy into the final file
      stats.rs                # ChromStats, GenomeStats, metadata CSV            R-STAT-*
      worker.rs               # process one chromosome
      run.rs                  # orchestration, scheduling                        R-RUN-*
      tmpdir.rs               # scratch directory                                R-RUN-5
      compare.rs              # parity comparison (used by tests and bam2df-compare)
  bam2df-cli/                 # binary `bam2df` (clap)
  bam2df-compare/             # binary `bam2df-compare` (parity tool, section 7.3)
tests/golden/synthetic/       # committed synthetic fixtures and Python outputs (section 7.1)
tools/golden/                 # Python scripts that make fixtures (run on the server)
tools/pickle_to_tsv.py        # mutation catalog converter (decision D5)
```

`target/` is already in `.gitignore`. Add `/golden/` (real-data fixtures, server only).

### 2.2 Crates

Pin exact versions in `Cargo.toml` when step 1 starts (use the current releases then) and commit
`Cargo.lock`. Check each crate's current API; the names below are what to look for, not a promise
of exact signatures.

| Need | Crate | Notes |
|---|---|---|
| BAM + BAI, FASTA + FAI | `noodles` (features `bam`, `sam`, `bgzf`, `csi`, `fasta`, `core`) | Pure Rust, no htslib build. Keep every noodles type inside `bamio.rs` and `reference.rs`, so API changes between versions touch two files. |
| bigWig | `bigtools` | Read-only features. |
| Parquet | `arrow-array`, `arrow-schema`, `parquet` (features `arrow`, `zstd`) | Not the `polars` crate: it is a heavy build and not needed. `parquet`'s `parquet-concat` binary is the reference for row-group copying. |
| npz | `zip` + `npyz` | Chromatin masks. |
| Hash | `blake2` | `Blake2bVar::new(8)`, R-ROW-14. |
| CLI | `clap` (derive) | Long flags with underscores, e.g. `#[arg(long = "output_file")]`. |
| Config | `serde`, `serde_json` | |
| ID parsing | `regex` | R-ID. |
| Errors | `anyhow` (binaries), `thiserror` (library) | |
| Hash maps | `rustc-hash` | Mate buffer, germline sets. |
| RNG | `rand`, `rand_chacha` | Match reservoir, D11. |
| Dev | `tempfile`, `proptest` | |

### 2.3 Data flow

```
main
 ├─ parse CLI, load and validate config                               R-CFG
 ├─ resolve sample/patient IDs                                         R-ID
 ├─ load shared inputs once → Arc<RunContext>                          R-IN, R-TRK-3..5
 │     region blacklists (merged IntervalSet per chromosome)
 │     germline SNV/indel sets per chromosome (+ reference check)
 │     mutation catalog per chromosome
 │     microsatellite and RepeatMasker lookups per chromosome
 │     BED regions per chromosome; paths of bigWigs and chromatin root
 ├─ make scratch dir                                                   R-RUN-5
 ├─ schedule chr1..chr22 on N worker threads, longest first            R-RUN-2
 │     worker(chrom):
 │        open own BAM reader, own BAM reader for depth, own FASTA, own bigWig readers
 │        load chromatin open bins for chrom (if enabled)
 │        for each region (whole chromosome, or each BED interval in order):
 │           fresh MateBuffer, fresh reservoir, fresh ChromStats       R-PAIR-1
 │           for read in query(region): filters → pairing → ReadView
 │               → observations → rows (all columns incl. tracks, chromatin,
 │                  patient_id, sample_id, tumor_mut_present) → RowBuilder
 │           flush RowBuilder every chunk_size rows → row group in part file
 │        returns (part file path, if any rows, and stats)              R-STAT-2
 ├─ concat part files in chr1..chr22 order into the output parquet    R-OUT-5
 └─ write metadata CSV                                                 R-STAT-3
```

Every row is complete when it leaves the worker. There is no second pass for chromatin and no
join for labels: both are lookups made while the row is built.

### 2.4 Core types (sketch)

```rust
pub struct RunContext {
    pub cfg: BamConfig,
    pub bam_path: PathBuf,
    pub reference_path: PathBuf,
    pub patient_id: String,
    pub sample_id: String,
    pub region_filters: FxHashMap<String, IntervalSet>,        // merged, D3
    pub germline_snv: FxHashMap<String, FxHashSet<(i64, u8, u8)>>,           // (start0, ref, alt)
    pub germline_indel: FxHashMap<String, FxHashSet<(i64, i64, Box<[u8]>, Box<[u8]>)>>,
    pub mutations: Option<FxHashMap<String, FxHashSet<(i32, Box<[u8]>, Box<[u8]>)>>>,
    pub microsat: Option<FxHashMap<String, AnnotationLookup>>,
    pub repeatmasker: Option<FxHashMap<String, AnnotationLookup>>,
    pub bed_regions: Option<BTreeMap<String, Vec<(i64, i64)>>>,
    pub replication_path: Option<PathBuf>,
    pub mappability_path: Option<PathBuf>,
    pub chromatin_root: Option<PathBuf>,
    pub schema: SchemaRef,                                      // fixed for the run, R-OUT-2
}

/// Decoded once per read that reaches pairing. Positions are 0-based.
pub struct ReadView {
    pub name: Box<[u8]>,
    pub flags: u16,
    pub ref_start: i64,           // reference_start
    pub ref_end: i64,             // reference_end (exclusive), R-READ-2
    pub cigar: Vec<(CigarOp, u32)>,
    pub seq_upper: Vec<u8>,       // query_sequence, upper-cased, includes soft clips
    pub qual: Vec<u8>,
    pub qa_start: i64,            // query_alignment_start, R-READ-3
    pub qa_end: i64,              // query_alignment_end
    pub qpos_at: Vec<i32>,        // index ref_pos - ref_start; -1 where no M/=/X base aligns
    pub ref_base_at: Option<Vec<u8>>, // MD-reconstructed reference base per qpos_at slot, R-READ-5
    pub md: Option<Box<[u8]>>,
    pub nm: Option<i64>,
    pub template_length: i64,
}

pub struct IntervalSet { starts: Vec<i64>, ends: Vec<i64> }  // sorted, merged, half-open

pub struct AnnotationLookup { starts: Vec<i64>, ends: Vec<i64>, labels: Vec<u32>, names: Arc<[Box<str>]> }
```

`RowBuilder` holds one Arrow builder per output column and a `push(&RowValues)` method; it is
turned into a `RecordBatch` with the run's schema at each flush.

### 2.5 Threads and memory

- Worker count `N = --threads`, else `SLURM_CPUS_PER_TASK`, else 1; capped at 22. Under gwf this
  is 12 (`src/gwf/template.py`, `cores="12"`).
- Each worker is single-threaded: single-threaded BGZF decoding, zstd compression on the worker's
  own thread. With 12 workers on 12 cores, extra threads would only compete (see the scheduling
  table in section 9).
- Per worker: the mate buffer (up to `max_mate_buffer` raw records), one row chunk, the depth cache
  (20 × 10 kb segments), the FASTA window (1 Mb), bigWig windows, the chromosome's chromatin open
  bins. Shared and read-only: everything in `RunContext`.
- Keep raw `bam::Record`s in the mate buffer; decode to `ReadView` only once a pair is complete.

---

## 3. Decisions

Defaults chosen for the port. The user can override any of them; record changes here.

| ID | Topic | Python behaviour | Rust behaviour |
|---|---|---|---|
| D1 | `use_paired_reads: false` | Fixed in Python (3.1) | Implement as Python does now: R-PAIR-10. |
| D2 | Optional input path given but file missing | Skipped with a warning (fixed in Python, 3.1): `[WARN] {label} path not found: {path}; continuing without it.`, with label `chromatin`, `replication timing`, `mappability`, `microsatellite`, `RepeatMasker` or `BED intervals`; blacklists keep their `[RawDataLoader:WARN]` lines (R-IN-1..3) | Same lines. |
| D3 | Overlapping intervals in region blacklists | Merged at load time (fixed in Python, 3.1) | Same: R-IN-5. |
| D4 | Narrow output dtypes (Int8 allele lengths, Int16 depth, ...) | Unverified (V8) | Keep the Python schema. Convert with checked casts; on overflow, fail with the column, value, chromosome and position. Widening is a step-8 option. |
| D5 | Mutation catalog format | Python pickle | Tab-separated file with header `chr  genomic_pos  ref  alt`. `tools/pickle_to_tsv.py` converts existing pickles. A path ending in `.pkl` fails with a hint to run the converter. |
| D6 | Scratch directory cleanup | A new subdirectory inside the first usable base, removed at exit (fixed in Python, 3.1) | Same: R-RUN-5. |
| D7 | Microsatellite or RepeatMasker path given, but no rows for a chromosome | Empty lookup, so the columns are present with defaults (fixed in Python, 3.1) | Same: columns present whenever the path is given; values `false` / `"no_microsatellite"` / `"no_repeat"`. |
| D8 | Chromatin mask with no open bins | Distance is NaN cast to uint16 (undefined, usually 0) | `dist_to_open = 65535`, `open = 0`. |
| D9 | Chromosome in `_VALID_CHROMS` missing from the BAM header | Crashes | Skipped with a `[WARN]`. |
| D10 | bigWig without the chromosome | Replication timing crashes; mappability gives NaN | NaN for both, one `[WARN]` per track and chromosome. |
| D11 | `n_match_sample > 0` reservoir | Seeded per region with `random.Random(f"{pre_sample_seed}:{chrom}:{start}:{end}")` (fixed in Python, 3.1), i.e. Mersenne Twister | ChaCha8 seeded from `pre_sample_seed`, chromosome and region. Reproducible, but a different sample from Python's; golden runs use `n_match_sample: 0`. |
| D12 | Read without an MD tag when mismatches or matches are extracted | pysam raises, the run dies | Fail with the read name. |
| D13 | `dist_to_indel_or_clip` | Removed (3.1) | Not computed. |


### 3.1 Bugs fixed in Python before the port

Fixed in the Python code first (commits `c1e5487` to `437da1c`), so the golden outputs show the
corrected behaviour and the Rust implements it directly. Tests: `tests/test_intervals.py`,
`tests/test_inputs.py` (run anywhere) and `tests/test_process_chromosome.py` (needs pysam, runs on
the server); all pass on the server.

| Bug | Fix |
|---|---|
| Single-read mode crashed with `NameError` (`highqual_overlap` unset), ignored its own quality positions, and used signed TLEN | Uses the read's quality positions within the region, `abs(TLEN)`, skips TLEN 0 (R-PAIR-10) |
| Blacklist lookup missed positions inside an earlier, longer overlapping interval | `prepare_blacklist_arrays` merges intervals |
| Stats of regions and chromosomes without rows were dropped from the metadata | `process_chromosome` and the worker always return stats |
| Overlapping BED intervals extracted the same reads twice | `load_bed_intervals` merges intervals |
| Microsatellite/RepeatMasker columns missing for a chromosome without annotation rows | Loaders return an empty lookup |
| `pre_sample_seed` unused; match subsampling not reproducible | Per-region seeded `random.Random` |
| Scratch cleanup could delete all of `$TMPDIR` | New subdirectory via `tempfile.mkdtemp`; only it is removed |
| Missing optional paths were skipped silently | `[WARN]` line (D2) |
| Label join relied on Polars casting Int64 catalog positions to the table's Int32 | Catalog built with explicit Int32 positions |
| Dead code: shadowed `_iter_mismatches`, duplicate replication-timing lookup, unused `dist_to_indel_or_clip` | Removed; no output change |
| README named the label column `tumor_label` and listed a nonexistent `somatic_label` | README and column ordering use `tumor_mut_present` |

---

## 4. CLI

`bam2df <bam_path> <reference_path> <config_path> [options]`, with exactly the Python flag names
so `src/gwf/template.py` only changes the command:

```
--output_file PATH                (required in Rust; Python crashes without it)
--mutation_path PATH              ("none", any case, means no catalog)
--genomic_blacklist_file PATH
--donor_blacklist_file PATH
--snp_blacklist_path PATH
--lowpower_blacklist_path PATH
--problematic_blacklist_path PATH
--bed_intervals_path PATH
--chromatin_path PATH
--replication_path PATH
--mappability_path PATH
--microsatellite_path PATH
--repeatmasker_path PATH
--sample_id ID
--patient_id ID
--verbose
--threads N                       (new; overrides SLURM_CPUS_PER_TASK)
--inputs-only                     (new, hidden; load and report inputs, then exit: used in step 1)
```

Exit code non-zero on any error. Log lines go to stdout with the same prefixes and formats as the
Python (`[INFO]`, `[BamProcessor:INFO]`, `[ChromosomeProcessor:INFO]`, `[RawDataLoader:WARN]`),
flushed per line, so logs can be compared (section 7.3).

---

## 5. Behaviour spec (the parity contract)

Coordinates: `pos0` is 0-based, `pos1` is 1-based. `[a, b)` is half-open.

### 5.1 Config (R-CFG)

Source: `config.json` key `"bam"`, `src/utility/config.py`, and `.get()` defaults across the code.

| Key | Required | Default | Validation |
|---|---|---|---|
| `context_flank` | yes | | integer ≥ 1 (Python crashes on 0, R-ROW-3) |
| `min_len`, `max_len` | yes | | integers |
| `min_base_quality` | yes | | non-negative integer; JSON `30.0`, `true`, `-1` rejected |
| `min_bq` | must be absent | | error containing `no longer supported` |
| `min_mapq` | no | 60 | integer |
| `bin_size_bp` | when chromatin is used | | integer ≥ 1 |
| `pre_sample_seed` | no | 0 | integer |
| `n_match_sample` | no | 0 | integer ≥ 0 |
| `mutation_types` | yes, non-empty | | each one of `match`, `mismatch`, `indel` |
| `use_paired_reads` | no | true | boolean; `false` selects single-read mode (R-PAIR-10) |
| `chunk_size` | no | 100000 | integer ≥ 1 |
| `max_mate_buffer` | no | 1000000 | integer ≥ 1 |

- **R-CFG-1** Error messages contain the same phrases the Python tests check:
  `no longer supported`, `missing required key`, `non-negative integer`.
- **R-CFG-2** Unknown keys: warn, do not fail.
- **R-CFG-3** `mutation_types` order is kept; it sets the order rows are emitted within a read.

### 5.2 Sample and patient IDs (R-ID)

Port `src/utility/sample.py` exactly, regexes included (`regex` crate syntax is compatible here):

- **R-ID-1** `get_sample_id(path)`: on the **file name**, `(C\d{5})_+(C\d+[A-Z]\d{5}D?)` → group 2;
  else on the **full path**, case-insensitive `(donor\d+)` → first letter upper-cased; else on the
  full path `(C\d+[A-Z]\d{5}D?)` → group 1; else error `Could not find sample ID in: {path}`.
- **R-ID-2** `get_patient_id(path)`: calls `get_sample_id` first (so it fails the same way); then file
  name pattern above → group 1; donor pattern; `/(C\d{5})(?:/|$)` on the full path; `[A-Z](\d{5})`
  on the sample ID → `C` + digits; else the sample ID.
- **R-ID-3** `resolve_sample_id`: the explicit ID if non-empty, else R-ID-1.
  `resolve_patient_id`: explicit ID if non-empty, else R-ID-2, and if that errors, the sample ID.
- Port the cases in `tests/test_sample_id.py` as unit tests.

### 5.3 Inputs (R-IN)

Valid chromosomes are exactly `chr1` … `chr22` (`_VALID_CHROMS`); rows on other names are dropped
after loading. No name normalization except in R-IN-6.

- **R-IN-1 Genomic blacklist** (`--genomic_blacklist_file`): tab-separated, **no header**,
  columns `chr start end [...]`. Missing file → `[RawDataLoader:WARN] genomic blacklist BED file not found at {path}, skipping.`
  Python would crash on header or comment lines; Rust may skip lines starting with `#`, `track`,
  `browser`.
- **R-IN-2 Donor blacklist** (`--donor_blacklist_file`): tab-separated **with a header line**;
  the first three columns are `chr start end` (Python renames exactly five columns
  `chr start end base count`; Rust uses the first three and ignores the rest). Missing file →
  `[RawDataLoader:WARN] donor blacklist file not found at {path}, skipping.`
- **R-IN-3 Low-power and problematic blacklists**: tab-separated, no header, columns `chr start end`
  (Python requires exactly three; Rust uses the first three). Missing file →
  `[RawDataLoader:WARN] Low-power blacklist missing for {patient_id}` /
  `[RawDataLoader:WARN] Problematic blacklist missing for {patient_id}`.
- **R-IN-4** All four region blacklists are `[start, end)` in 0-based coordinates and apply to `pos0`.
- **R-IN-5** Build one merged `IntervalSet` per chromosome from the union of all four (D3); overlapping
  and touching intervals merge, as `merge_intervals` in `src/utility/io.py` does.
  `contains(pos0)`: `i = partition_point(|s| s <= pos0) - 1; i >= 0 && pos0 < ends[i]`.

- **R-IN-6 Germline calls** (`--snp_blacklist_path`, port `src/utility/variant_blacklist.py`):
  - A given but missing file is an error (`SNP blacklist not found: {path}`).
  - Tab-separated with header; all fields read as strings. Header names normalized as
    `strip → lower → strip leading '#'`. Required: `chr`, `start`, `ref`, `alt`; otherwise error
    `SNP blacklist {source} is missing columns [...]; found [...]`.
  - Per data row (1-based index in messages: `{source} data row {n}`): `start` parsed like Python
    `int()` (surrounding whitespace and a leading `+` allowed) else error `... is not an integer`;
    negative → `... is negative`. `ref` = strip + upper; must match `^[ACGTN]+$` else
    `REF ... is not an ACGTN sequence`. Chromosome: strip; prefix `chr` if missing.
  - `alt` split on `,`; each strip + upper; `*` or `.` → count `skipped_non_sequence_allele`;
    not ACGTN → error `ALT ... is not an ACGTN sequence`; equal to ref → error `ALT equals REF (...)`.
  - `len(ref) == len(alt) == 1` → SNV key `(start0, ref, alt)`; equal length > 1 →
    `skipped_multi_base_substitution`; one a prefix of the other → indel key
    `(start0, start0 + len(ref), ref, alt)`; otherwise `skipped_complex_indel`.
  - No SNV and no indel → error containing `no usable`.
- **R-IN-7** Keep only valid chromosomes, then log exactly:
  `[INFO] SNP blacklist {path}: {'snv': N, 'indel': N, 'skipped_multi_base_substitution': N, 'skipped_complex_indel': N, 'skipped_non_sequence_allele': N}; kept on processed chromosomes: snv={n:,} indel={n:,}`
  (Python dict repr; `{:,}` is thousands separators with commas).
- **R-IN-8 Reference check:** iterate chromosomes in **string** sort order (`chr1, chr10, chr11, …, chr2, …`),
  and within each, SNV keys in tuple order `(start0, ref, alt)`; check at most 10,000. Compare the
  FASTA base at `start0`, upper-cased, with `ref`. If checked > 0 and mismatched / checked > 0.05:
  error `SNP blacklist {path}: REF disagrees with {reference_path} at {m:,} of {n:,} checked SNVs. START must be 0-based on the same assembly as the reference.`
- **R-IN-9** Port all of `tests/test_variant_blacklist.py` and `tests/test_blacklist_lookup.py` as unit tests.

- **R-IN-10 BED regions** (`--bed_intervals_path`): skip empty lines and lines starting with `#`;
  split on tab; fewer than 3 fields → error `BED parse error at line {n}: expected >=3 columns`;
  `end <= start` → error `BED parse error at line {n}: end must be > start, got {chrom}:{start}-{end}`.
  Per chromosome, sort and **merge** overlapping or touching intervals (as R-IN-5). With BED mode
  on, chromosomes without intervals are not processed.
- **R-IN-11 Mutation catalog** (D5): TSV with header `chr genomic_pos ref alt`. Keys
  `(chr, genomic_pos, ref, alt)`, duplicates collapse. `--mutation_path` absent or `none` →
  `[INFO] No mutation file provided, continuing without mutation index.`; given but missing →
  `[INFO] No mutation file found at {path}, continuing without it.`

### 5.4 Reading the BAM (R-READ, R-FILT, R-PAIR)

Per worker, per region `[start, end)` (whole chromosome `[0, reference_length)`, or each BED
interval): query the BAM index for records overlapping the region, in file order.

**Read fields** (pysam meaning):

- **R-READ-1** Flags: paired `0x1`, unmapped `0x4`, reverse `0x10`, read1 `0x40`, read2 `0x80`,
  secondary `0x100`, qcfail `0x200`, duplicate `0x400`, supplementary `0x800`.
- **R-READ-2** `reference_end` = `reference_start` + sum of lengths of CIGAR ops M, D, N, =, X.
- **R-READ-3** `query_alignment_start` = total length of leading S ops (H ops skipped);
  `query_alignment_end` = sequence length − total length of trailing S ops;
  `query_alignment_length` = end − start.
- **R-READ-4** Aligned pairs (pysam `get_aligned_pairs(matches_only=True)`): walk the CIGAR;
  M/=/X yield `(qpos, rpos)` and advance both; I and S advance qpos; D and N advance rpos;
  H and P advance nothing.
- **R-READ-5** Reference base per aligned pair (pysam `with_seq=True`): the read's own base,
  unless the MD tag marks that reference position as a mismatch, in which case it is the MD letter.
  Parse MD as tokens: digits (match run), a letter (mismatch, consumes one reference position),
  `^` + letters (deletion, consumes reference positions, no aligned pair). MD walks reference
  positions over M/=/X and D only. A malformed or inconsistent MD is an error naming the read
  (pysam raises too). A pair is a **mismatch** iff `upper(read_base) != upper(ref_base)`; this means
  an MD letter equal to the read base counts as a match, as in Python.
- **R-READ-6** Missing qualities: BAM quality bytes all `0xFF` (pysam `query_qualities is None`).
  Empty sequence (`*`) means `query_sequence` is None.

**Read filters**, in this order; any failure increments `n_skipped` and drops the read:

- **R-FILT-1** Reject unmapped, secondary, supplementary, duplicate, qcfail, or not paired.
  Reject missing or empty qualities. Reject `mapq < min_mapq`.
- **R-FILT-2** Type predicate: keep the read if **any** type in `mutation_types` holds:
  - `indel`: the CIGAR has an I or D op.
  - `mismatch`: NM tag present and `NM − (total I + D length) > 0`. No NM → false.
  - `match`: no CIGAR → false; `aligned = max(0, query_alignment_length − total I length)`;
    `aligned == 0` → false; no NM → **true**; else `aligned − max(0, NM − (I + D length)) > 0`.
  - NM may be stored as any BAM integer type.

**Pairing** (paired mode, the default; single-read mode is R-PAIR-10):

- **R-PAIR-1** A fresh mate buffer, reservoir and stats per region.
- **R-PAIR-2** Look up the read's name in the buffer and remove it. If absent: run eviction
  (R-PAIR-3), insert the read, push its name to the FIFO, and continue with the next read.
- **R-PAIR-3** Eviction, run before each insert: `while buffer.len() >= max_mate_buffer`: if the
  FIFO is empty, add `buffer.len()` to `n_evicted`, clear the buffer, stop; else pop the oldest
  name from the FIFO and remove it from the buffer, adding 1 to `n_evicted` only if it was present.
  Names of reads that were paired stay in the FIFO as stale entries; reproduce this exactly
  (it is what makes eviction counts match).
- **R-PAIR-4** If the read is read1 and the buffered one read2 (or the reverse), name them
  `read1`/`read2`; otherwise `n_skipped += 1` and drop both.
- **R-PAIR-5** Overlap: `max(ref_start)` to `min(ref_end)` of the two; empty → skip. Clip to the
  region: `[max(ovl_start, start), min(ovl_end, end))`; empty → skip.
- **R-PAIR-6 High-quality positions:** every `rpos` in the clipped overlap where **both** reads have
  an aligned pair (R-READ-4) and both qualities at those qpos are `>= min_base_quality`. Store as a
  bitset over the clipped overlap. Empty → skip.
- **R-PAIR-7 Fragment:** the two must be on opposite strands, else invalid; `fwd` is the
  non-reverse one, `rev` the reverse one; `frag_len = rev.ref_end − fwd.ref_start`; invalid if
  `frag_len <= 0` or outside `[min_len, max_len]` (inclusive). Invalid → `n_invalid_fragments += 1`
  **and** `n_skipped += 1`, skip.
- **R-PAIR-8 Fragment end contexts** with `k = context_flank`: `left` = the read with the smaller
  `ref_start` (read1 on a tie), `right` = the other. `start_ctx` = first k bases of `left`'s upper-cased
  full query sequence (soft clips included), `end_ctx` = last k bases of `right`'s.
- **R-PAIR-9** The **driver** is `read2`. Only its observations are extracted, against the pair's
  clipped overlap and high-quality positions. Then `n_total += n_opp` (R-EXT-4) and `n_reads += 1`.
- **R-PAIR-10 Single-read mode** (`use_paired_reads: false`): no mate buffer; each read that passes
  R-FILT is extracted on its own, in place of R-PAIR-2..9:
  `frag_len = |template_length|`; invalid (`n_invalid_fragments += 1`, `n_skipped += 1`) if 0 or
  outside `[min_len, max_len]`. The span is `[max(ref_start, start), min(ref_end, end))`; empty →
  skip. High-quality positions are the read's aligned positions in the span with quality
  `>= min_base_quality`; none → skip. `start_ctx`/`end_ctx` are the first and last k bases of the
  read's own upper-cased query sequence (soft clips included). The read is the driver, with the span
  as the overlap, so both mates of a pair are extracted. `n_total += n_opp`, `n_reads += 1`.

### 5.5 Extraction (R-EXT)

Run one extractor per entry of `mutation_types`, in config order; a type contributes rows and
opportunities only if its `n_opp > 0`. Let `HQ` be the high-quality set, `[os, oe)` the clipped
overlap, and `filtered(pos0)` the merged region blacklist (R-IN-5) for the chromosome.

- **R-EXT-1 Indels.** Walk the driver's CIGAR with `ref_pos0 = ref_start`, `qpos = 0`:
  - M/=/X: advance both by len.
  - I: `anchor = ref_pos0 − 1`. If `anchor < ref_start`, advance qpos and skip. Otherwise:
    `ins = seq_upper[qpos .. qpos+len]`, `a = upper(FASTA base at anchor)`; candidate
    `(qpos, anchor, "I", ref=a, alt=a+ins)`; advance qpos.
  - D: `anchor = ref_pos0 − 1`. If `anchor < ref_start`, advance ref_pos0 and skip. Otherwise:
    `a = upper(FASTA base at anchor)`, `del = upper(FASTA bases [ref_pos0, ref_pos0+len))`;
    candidate `(qpos, anchor, "D", ref=a+del, alt=a)`, where qpos is the query position after the
    deletion; advance ref_pos0.
  - N: advance ref_pos0. S: advance qpos. H, P: nothing.
  - Keep candidates whose **anchor** is in HQ (there is no separate overlap check), then drop
    those with `filtered(anchor)`. `n_opp` = number left.
  - For each: skip if `ref == alt`; skip if `(anchor, anchor + len(ref), ref, alt)` is a germline
    indel key; build a row with `pos1 = anchor + 1`, `query_pos = qpos`, `other_err = 0`.
- **R-EXT-2 Mismatches.** Requires MD (D12). `other_err` = number of MD letters outside `^` blocks
  (port `count_other_errors_in_read`). `n_opp` = number of positions in HQ (all of which lie in
  `[os, oe)`) with `!filtered`. Rows: for each aligned pair `(qpos, rpos)` of the driver that is a
  mismatch (R-READ-5) with `os <= rpos < oe`, `rpos ∈ HQ`, `!filtered(rpos)`:
  `alt = seq_upper[qpos]`, `ref = upper(ref_base)`; skip if equal; skip if `(rpos, ref, alt)` is a
  germline SNV key; build a row with `obs_type "S"`, `pos1 = rpos + 1`, `other_err`.
- **R-EXT-3 Matches.** Requires MD. For each aligned pair of the driver with `os <= rpos < oe`,
  read base equal to reference base (case-insensitive), `rpos ∈ HQ`, `!filtered(rpos)`: one
  opportunity and one row, `obs_type "M"`, `alt = seq_upper[qpos]`, `ref = upper(ref_base)`,
  `other_err = 0`. No germline check.
- **R-EXT-4** `n_opp` for the read is the sum over types with `n_opp > 0`. It is counted before the
  germline filter, the `ref == alt` check and R-ROW-1, so rows can be fewer than opportunities.
- **R-EXT-5 Row routing:** for every row built, update stats by `obs_type`: `S` → `n_mismatches`,
  `M` → `n_matches`, `I`/`D` → `n_indels`. If `obs_type == "M"` and `n_match_sample > 0`, the row
  goes through reservoir sampling (D11) instead of the chunk; the reservoir is written after the
  region's last chunk. Otherwise it goes to the chunk; flush when the chunk reaches `chunk_size`.

### 5.6 Row features (R-ROW)

`ctx(pos1, flank)` = FASTA bases `[max(pos1 − flank − 1, 0), +2·flank+1)`, clipped at the contig
end, **case preserved** (R-REF-1).

- **R-ROW-1** `reference_ctx = ctx(pos1, flank)`. If shorter than `2·flank+1`, drop the row
  (the stats in R-EXT-5 are not updated for it).
- **R-ROW-2** Near a contig start the context is shifted, not centred (quirk Q2). Reproduce it.
- **R-ROW-3** `trinucleotide_ctx = reference_ctx[flank−1] + reference_ctx[flank] + reference_ctx[flank+1]`.
- **R-ROW-4** `local_gc = (count of 'G' + count of 'C') / (2·flank+1)`, uppercase letters only (Q1).
- **R-ROW-5** `local_complexity_1`, `local_complexity_2`: Shannon entropy (log2) of the bytes of
  `reference_ctx` and of its overlapping 2-mers, case-sensitive. To match Python's float sum:
  accumulate terms `p · log2(p)` in f64 in **first-occurrence order** of each symbol (Python
  `Counter` order), with `p = count / n` (`n − 1` for 2-mers), then negate. A 1-base context gives
  `h2 = 0`.
- **R-ROW-6** `n_other_err = other_err − 1` for `S` rows, else `0`.
- **R-ROW-7** `read_depth` = depth at `pos0 = pos1 − 1` (R-DEPTH).
- **R-ROW-8** `base_qual_site = qual[query_pos]`.
- **R-ROW-9** `dist_to_end = min(query_pos − qa_start, qa_end − 1 − query_pos)`.
- **R-ROW-10** `homopolymer_run_len`: `w = ctx(pos1, 50)` (101 bases, shifted near the contig start);
  `c = w[50]`; count equal bytes (case-sensitive) leftwards from index 49 and rightwards from 51;
  result `left + 1 + right`.
- **R-ROW-11** `read_index = query_pos − qa_start + 1`; `seq_length = query_alignment_length`;
  `fragment_len` from R-PAIR-7 (R-PAIR-10 in single-read mode); `first_in_pair = 1 if driver is read1
  else 0` (always 0 in paired mode, Q3); `strand = "rev"` if the driver is reverse else `"fwd"`.
- **R-ROW-12** `obs_alt_len = len(alt)`, `obs_ref_len = len(ref)`, `len_delta = alt_len − ref_len`.
- **R-ROW-13** `genomic_pos = pos1`; `chr` = chromosome name; `start_ctx`, `end_ctx` from R-PAIR-8
  (R-PAIR-10 in single-read mode).
- **R-ROW-14** `qname` = BLAKE2b with digest size 8 (unkeyed) of the read name bytes, read as a
  little-endian u64. `Blake2bVar::new(8)` gives this; it is not a truncated 64-byte digest. Check
  against the vectors from step 0.

### 5.7 Reference access (R-REF)

- **R-REF-1** Raw FASTA bytes, case preserved (hg38 is soft-masked). Upper-case only where a rule
  says so (indel alleles).
- **R-REF-2** The Python 1 Mb sliding cache is transparent: its results equal a direct fetch of the
  same range clipped at the contig end. Rust keeps a window cache per worker for speed:
  on a miss, load `[max(0, pos1 − 500000), +1000000)`.

### 5.8 Read depth (R-DEPTH)

Python `_get_read_depth` → `pysam.AlignmentFile.count_coverage` with a callback. Verify V1 before
implementing.

- **R-DEPTH-1** Segments of 10,000 bp: `seg = floor(pos0 / 10000) · 10000`. A cache of at most 20
  segments per worker, evicting the **oldest inserted** (FIFO, not LRU); clear on a new contig.
- **R-DEPTH-2** Computing a segment `[seg, seg + 10000)`, stop clipped to the contig length: for every
  record overlapping it (a second reader, not the main one), keep the record only if it passes the
  depth filter: R-FILT-1 **and** `|template_length| != 0` and `min_len <= |template_length| <= max_len`.
  For each aligned pair (R-READ-4) with `rpos` in the segment, quality `>= 15`, and read base
  exactly `A`, `C`, `G` or `T`: add 1 at `rpos`.
- **R-DEPTH-3** Depth is the sum over the four bases. Both mates of an overlapping pair count (Q8).
  A position past the clipped segment end has depth 0.

### 5.9 Tracks (R-TRK)

- **R-TRK-1 Replication timing** (column only if the path exists): bigWig value over
  `[pos0, pos0+1)`; missing or NaN → NaN (a float NaN, not null).
- **R-TRK-2 Mappability** `umap_score`: same, NaN when missing. Values are the bigWig's f32.
  Python reads per row; Rust caches a window of intervals per worker (e.g. 1 Mb ahead of the
  current position), with no change in values.
- **R-TRK-3 Microsatellites** (`--microsatellite_path`, UCSC simpleRepeat-like): skip empty lines and
  lines starting with `#`; split on whitespace; fewer than 5 fields → skip; `chrom = f[1]`,
  `start = f[2]`, `end = f[3]`, `motif = f[4]`. Per chromosome, **stable** sort by start.
- **R-TRK-4 RepeatMasker** (UCSC rmsk.txt): same skipping; fewer than 13 fields → skip;
  `chrom = f[5]`, `start = f[6]`, `end = f[7]`, `class = f[11]`. Python reads only the **first
  contiguous block** of rows for each chromosome and stops at its end; when loading all chromosomes
  in one pass, ignore later rows of a chromosome whose block already ended. Stable sort by start.
- **R-TRK-5 Lookup** (both): `i = partition_point(|s| s <= pos0) − 1`; hit iff `i >= 0 && pos0 < ends[i]`.
  Not merged: it checks only that one interval, as in Python (Q11). A chromosome without rows has
  an empty lookup, so the columns are still written (D7). Microsatellite columns:
  `in_microsatellite`, `microsatellite_motif` (`"no_microsatellite"` on a miss). RepeatMasker:
  `in_repeat_region`, `repeat_class` (`"no_repeat"`).
- **R-TRK-6 Chromatin masks:** `{root}/chromatin_masks/{ATAC|DNase}/sparse_overlaps_by_chromosome/consensus/{chrom}.npz`
  (scipy `save_npz`). Missing file → error `Missing chromatin mask: {path}`. Read `format`, `shape`
  and the arrays; implement the format(s) V5 finds (`csr` expected: `data`, `indices`, `indptr`).
  Open bins = positions of nonzero `data` entries: the row index if `shape[1] == 1`, the column index
  if `shape[0] == 1`, otherwise the row index; sorted.
- **R-TRK-7** `bin = genomic_pos // bin_size_bp`, using the **1-based** `genomic_pos` (Q7).
- **R-TRK-8** With `open` the sorted bins and `j = lower_bound(open, bin)` clipped to `[0, n−1]`:
  `{atac,dnase}_open = 1 if open[j] == bin else 0` (Int8);
  `{atac,dnase}_dist_to_open = min(min(|bin − open[clip(j−1)]|, |bin − open[j]|) · bin_size_bp, 65535)` (UInt16).
  Empty `open`: D8.

### 5.10 Identity and label columns (R-LBL)

- **R-LBL-1** `patient_id`, `sample_id`: constant strings per run.
- **R-LBL-2** `tumor_mut_present` (Int8): 1 if `(chr, genomic_pos, obs_ref, obs_alt)` is in the
  catalog, else 0; always 0 without a catalog. The column is always present.

### 5.11 Output (R-OUT)

- **R-OUT-1 Column order and types**, matching `order_columns_by_schema`:

  | Column | Type | | Column | Type |
  |---|---|---|---|---|
  | `qname` | UInt64 | | `local_complexity_1` | Float32 |
  | `chr` | String | | `local_complexity_2` | Float32 |
  | `genomic_pos` | Int32 | | `n_other_err` | Int8 |
  | `obs_type` | Categorical | | `base_qual_site` | Int8 |
  | `obs_alt` | String | | `dist_to_end` | Int16 |
  | `obs_ref` | String | | `homopolymer_run_len` | Int16 |
  | `obs_alt_len` | Int8 | | `read_depth` | Int16 |
  | `obs_ref_len` | Int8 | | `replication_timing`* | Float32 |
  | `len_delta` | Int8 | | `umap_score`* | Float32 |
  | `strand` | String | | `atac_open`* | Int8 |
  | `first_in_pair` | Int8 | | `atac_dist_to_open`* | UInt16 |
  | `read_index` | Int16 | | `dnase_open`* | Int8 |
  | `fragment_len` | Int16 | | `dnase_dist_to_open`* | UInt16 |
  | `seq_length` | Int16 | | `in_microsatellite`* | Boolean |
  | `trinucleotide_ctx` | String | | `microsatellite_motif`* | String |
  | `start_ctx` | String | | `in_repeat_region`* | Boolean |
  | `end_ctx` | String | | `repeat_class`* | String |
  | `reference_ctx` | String | | `patient_id` | String |
  | `local_gc` | Float32 | | `sample_id` | String |
  | | | | `tumor_mut_present` | Int8 |

  \* Present only when the corresponding input path was given and exists.
- **R-OUT-2 Arrow types:** the exact physical Arrow types (e.g. `large_string` versus `string`, and the
  dictionary encoding of `obs_type`) must equal the schema dumped from the Python output in step 0
  (V2). Expected: `large_string` strings and `dictionary<values=large_string, indices=uint32>` for
  `obs_type`. All fields nullable. Follow the dump if it differs.
- **R-OUT-3** Narrow integers: checked conversion (D4).
- **R-OUT-4** Parquet: zstd compression; one row group per flushed chunk (≤ `chunk_size` rows).
- **R-OUT-5** Part files are written by each worker into the scratch directory. The final file is
  their concatenation in `chr1 … chr22` order (regions in BED order within a chromosome), made by
  copying row groups without decoding (see `parquet-concat` in the `parquet` crate). Then remove the
  scratch directory.

### 5.12 Stats and metadata (R-STAT)

- **R-STAT-1** `ChromStats` counters: `n_reads, n_mismatches, n_matches, n_indels, n_total,
  n_skipped, n_evicted, n_invalid_fragments`.
- **R-STAT-2** Every processed region contributes its stats to its chromosome, and every processed
  chromosome to the genome, whether or not it produced rows. For each processed chromosome, log
  `[ChromosomeProcessor:INFO] Finished {chrom} in {s:.1f}s ({m:.2f} min) — [{chrom}] reads={:,}, mismatches={:,}, matches={:,}, indels={:,}, opportunities={:,}, skipped={:,}, evicted={:,}, invalid={:,}`.
  If no chromosome produced rows: error `No features extracted from BAM {bam_path}`.
- **R-STAT-3** Metadata CSV at `{output_dir}/{output_stem}_metadata.csv`, columns
  `sample_id, chrom, coverage, mismatches, matches, indels, mismatch_load, indel_load, beta`,
  rows `chr1 … chr22` (zeros for absent chromosomes) then `total`:
  `coverage = n_total`, `mismatch_load = mismatches / max(coverage, 1)`,
  `indel_load = indels / max(coverage, 1)`, `beta = matches / max(coverage − mismatches, 1)`.
  Float formatting may differ from polars; compared numerically.

### 5.13 Run (R-RUN)

- **R-RUN-1** Startup order: config → output directory creation → mutation catalog → track paths
  (exist-or-skip, D2) → IDs → blacklists (with germline check) → BED regions → extraction → output
  → metadata.
- **R-RUN-2 Scheduling:** sort the chromosomes to process by reference length, longest first; N
  workers pull the next one from a shared atomic index.
- **R-RUN-3** A worker that fails makes the run fail (no partial output file).
- **R-RUN-4** `--verbose` controls the same log lines as in Python.
- **R-RUN-5 Scratch directory:** first writable of `$TMPDIR`, `/tmp/$SLURM_JOB_ID`, `$TMP`, `/tmp`;
  create a unique subdirectory inside it (D6); remove it at exit, also on error.

---

## 6. Python quirks reproduced for parity

The port reproduces these. Each is a candidate for a deliberate fix in step 8, which will change the
output. Q5 and Q6 were bugs and are fixed in Python (section 3.1); the IDs are not reused.

| ID | Quirk |
|---|---|
| Q1 | Reference context is not upper-cased: `trinucleotide_ctx`, `reference_ctx` carry soft-mask case; `local_gc` counts only uppercase G/C; entropy and homopolymer are case-sensitive. |
| Q2 | Contexts and the homopolymer window are shifted, not centred, within `flank+1` (resp. 51) bases of a contig start. |
| Q3 | `first_in_pair` is always 0, because the driver is always read2. |
| Q4 | Indels are kept on the anchor base being high-quality, without an overlap-range check. |
| Q7 | Chromatin bins use the 1-based position. |
| Q8 | `read_depth` counts both mates of an overlapping pair. |
| Q9 | `start_ctx`/`end_ctx` include soft-clipped bases. |
| Q10 | Mismatch opportunities count every high-quality position, not only positions where a mismatch could be emitted after germline filtering. |
| Q11 | Microsatellite and RepeatMasker lookups check only the last interval starting at or before the position, so a position inside an earlier, longer interval (for example an L1 containing an Alu) is reported as outside. The blacklist lookup had the same flaw and was fixed by merging (3.1); annotations carry labels, so the fix here needs an interval query that picks a label. |

---

## 7. Verification

### 7.1 Golden fixtures

Two sets, both made by the **Python** code at the parity commit, on the server, in step 0.

**S, synthetic (committed under `tests/golden/synthetic/`)**, made by `tools/golden/make_synthetic.py`
using only pysam, pyBigWig, numpy and scipy (pysam bundles `samtools`: use `pysam.calmd`,
`pysam.sort`, `pysam.index`, `pysam.faidx`, so no separate samtools is needed):

- FASTA with all of `chr1 … chr22` (chr1 40 kb, chr2 20 kb, the rest 2 kb) plus `chrX` (5 kb);
  random bases with some lowercase soft-masked stretches and some homopolymer runs. The BAM header
  lists every contig.
- Reads built from the reference with designed edits; MD and NM filled in by `pysam.calmd`.
  Cases, at least:
  1. clean overlapping pair (rows only in match mode);
  2. SNV in the overlap, both mates high quality → row;
  3. SNV in the overlap with the mate's base quality low there → no row, fewer opportunities;
  4. SNV in the driver outside the overlap → no row;
  5. SNV only in read1 → no row (driver is read2);
  6. insertion and deletion in the overlap; a read with several mismatches (`n_other_err`);
  7. insertion right after a leading soft clip (anchor before `ref_start`, skipped);
  8. soft clips at both ends with an SNV near the end (`read_index`, `dist_to_end`);
  9. pairs in both orientations (read1 forward and read1 reverse);
  10. fragment longer than `max_len`; same-strand mates (both invalid);
  11. low MAPQ, duplicate, qcfail, secondary, supplementary reads;
  12. a read whose mate is absent (stays buffered);
  13. pairs within 51 bp of a contig start (Q2) and within `flank` of a contig end (row dropped);
  14. SNVs in lowercase reference stretches (Q1) and in a homopolymer run;
  15. read base `N` at a mismatch;
  16. a stack of extra reads over one site for depth: some low MAPQ, `tlen 0`, duplicates, bases
      with quality < 15;
  17. reads on `chrX` (ignored).
- Inputs: genomic BED, donor TSV, low-power and problematic files (including a nested and a
  touching pair of intervals, which both implementations merge), a germline TSV (one exact SNV
  match, one same-site different ALT, one exact indel), replication and mappability bigWigs with a
  gap (NaN), microsatellite and RepeatMasker text files (with one nested interval, Q11) and one
  chromosome with reads but no annotation rows (D7), ATAC and DNase `.npz` masks, a mutation catalog
  as pickle and as TSV (D5) containing some emitted SNVs and an indel, a BED regions file with two
  overlapping intervals and one separate interval.
- Config: `chunk_size: 50` (several chunks), `min_mapq: 20`, `n_match_sample: 0`.

Runs (`tools/golden/run_python_golden.sh synthetic`), each saving the parquet, the metadata CSV,
the log, and `schema.txt` (`pyarrow.parquet.read_schema`):

| Run | Inputs | `mutation_types` |
|---|---|---|
| S1 core | BAM, FASTA, config | mismatch, indel |
| S2 filters | S1 + all four region blacklists + germline | mismatch, indel |
| S3 full | S2 + all tracks + chromatin + mutation catalog | mismatch, indel |
| S4 bed | S3 + BED regions | mismatch, indel |
| S5 match | S3 + BED regions | match, mismatch, indel |
| S6 evict | S1 with `max_mate_buffer: 2` | mismatch, indel |
| S7 overflow probe | a read with a ≥ 128-base insertion | mismatch, indel |
| S8 single-read | S3 with `use_paired_reads: false` | mismatch, indel |

S7 only records what Python does on Int8 overflow (V8); if it crashes, save the error text.

The script also writes `tests/golden/synthetic/vectors.json` with reference values computed in
Python: `qname` hashes for several names (including an empty name), `calculate_local_complexity`
and GC for a few contexts (mixed case, `N`, all one base), `count_other_errors_in_read` for MD
strings with `^` blocks, and a few `get_sample_id`/`get_patient_id` results.

**R, real data (server only, under `/golden/`, gitignored)**, made by
`tools/golden/make_real_subset.sh` and `run_python_golden.sh real`:

- A subset BAM from one real BAM, e.g. `chr21:20,000,000-22,000,000` and
  `chr22:30,000,000-32,000,000` (all header contigs kept), indexed. Shrink the regions if Python
  takes more than about an hour.
- Real reference, tracks, blacklists for that patient, real config with `n_match_sample: 0`.
- Runs R1–R5 as S1–S5, with the BED regions for R4/R5 a few intervals of ≤ 20 kb inside the subset
  (match mode emits a row per matched base).
- Record wall time and peak memory (`/usr/bin/time -v`) per run.
- **Full-BAM reference** for step 6: one complete BAM through Python with the production config and
  inputs (12 cores), output kept under `/golden/full/`.

### 7.2 Things to verify in step 0

| ID | Check | How |
|---|---|---|
| V1 | `count_coverage` semantics: quality threshold 15, only `ACGT`, M/=/X pairs, callback | Read `count_coverage` in `pysam/libcalignmentfile.pyx` at the tag of the installed pysam version; note anything that differs from R-DEPTH. |
| V2 | Arrow schema of the Python output | `schema.txt` from each run. |
| V3 | Labels are set | S3 has some `tumor_mut_present == 1`, at the catalog's records. |
| V4 | Chromatin columns build (float arrays into Int8/UInt16) | S3 completes. |
| V5 | npz format of the real masks | `tools/golden/inspect_inputs.py`: print `format`, `shape`, dtypes for a few masks. |
| V7 | Soft-masking in the real FASTA | Same script: fraction of lowercase bases on chr21. |
| V8 | Python on Int8/Int16 overflow | S7. |
| V9 | Versions | Same script: pysam, polars, pyarrow, numpy, git commit. Saved to `versions.txt`. |

### 7.3 Comparison tool: `bam2df-compare`

```
bam2df-compare parquet EXPECTED ACTUAL [--ignore COL ...] [--chrom CHR] [--max-examples 10]
bam2df-compare fingerprint EXPECTED ACTUAL          # per-chromosome, for full-BAM outputs
bam2df-compare csv EXPECTED ACTUAL                  # metadata CSV, numeric with tolerance
bam2df-compare logs EXPECTED.log ACTUAL.log         # "Finished chrN" stat lines and the SNP blacklist line
```

- **Schema:** names, order, Arrow types and nullability must be equal (ignored columns excluded).
- **Rows:** equal counts; then sort both by all non-float columns in schema order, then by float
  columns, and compare row by row.
- **Floats:** NaN equals NaN; otherwise equal if `|a − b| <= 1e-6 · max(1, |a|)` or within 2 f32 ULPs.
- **Report:** per column, the number of differing rows and the first examples with
  `chr, genomic_pos, obs_type, obs_ref, obs_alt, qname`. Exit 0 only when everything matches.
- **Fingerprint mode:** stream both files; per chromosome, row count plus order-independent sum and
  xor of 64-bit hashes of each row (floats rounded to 6 significant digits). A chromosome that
  differs is then compared with `parquet --chrom`.
- The same comparison code runs in the Rust integration tests against the committed S outputs.

---

## 8. Steps

Every step ends with a commit, once the user approves it. "Server test" lists what the user runs;
"Pass" is what must hold. Common server preamble (the `git pull` is the only frontend command):

```bash
cd ~/bam2df && git pull
conda activate bam2df
srun --account "$ACCOUNT" -c 4 --mem 8g --time 01:00:00 \
    bash -c 'cargo fmt --check && cargo build --release && cargo test --release'
```

The other server commands of a step run the same way, wrapped in `srun` with `-c`, `--mem` and
`--time` sized to the job; for brevity they are listed below without the wrapper.

### Step 0: Golden fixtures (Python only)

Build: `tools/golden/make_synthetic.py`, `tools/golden/run_python_golden.sh`,
`tools/golden/make_real_subset.sh`, `tools/golden/inspect_inputs.py`, `tools/pickle_to_tsv.py`;
add `/golden/` to `.gitignore`.

Server test:
```bash
python tools/golden/make_synthetic.py tests/golden/synthetic
bash tools/golden/run_python_golden.sh synthetic
python tools/golden/inspect_inputs.py --config golden/real_inputs.env > golden/inspect.txt
bash tools/golden/make_real_subset.sh golden/real_inputs.env
bash tools/golden/run_python_golden.sh real
```
`golden/real_inputs.env` is a file the user writes on the server with the real paths (BAM,
reference, tracks, blacklists, regions); a template `tools/golden/real_inputs.env.example` with
placeholders is committed.

Pass: S1–S6, S8 and R1–R5 complete; S7 result recorded; `vectors.json` written; the user commits
`tests/golden/synthetic/` (synthetic only) and pastes `inspect.txt`, the V1 notes and the run
timings back. Record the V answers in this document (section 7.2) before step 1.

### Step 1: Workspace, config, CLI, IDs, inputs, compare tool

Build: the workspace; `config.rs`, `ids.rs`, `inputs/*`, `tracks/annotation.rs` loaders,
`tmpdir.rs`, the CLI with `--inputs-only` (loads everything per R-RUN-1, prints the R-IN-7 line and a
summary of interval and annotation counts per chromosome, exits 0), `compare.rs` and
`bam2df-compare`. Unit tests: ports of `tests/test_config.py`, `test_sample_id.py`,
`test_variant_blacklist.py`, `test_blacklist_lookup.py`, `test_intervals.py` and the input cases of
`test_inputs.py` (empty annotation lookups, scratch directory, missing-path warning); the
`vectors.json` ID cases; IntervalSet merge and lookup; annotation lookup ties (stable sort);
RepeatMasker first-block rule.

Server test:
```bash
./target/release/bam2df-compare parquet tests/golden/synthetic/S3/out.parquet tests/golden/synthetic/S3/out.parquet
./target/release/bam2df <R3 arguments> --inputs-only > golden/R3_rust_inputs.log
grep "SNP blacklist" golden/R3/python.log golden/R3_rust_inputs.log
```
Pass: tests green; the compare tool reports a file identical to itself; the SNP blacklist lines are
identical.

### Step 2: Core extraction without depth or tracks

Build: `bamio.rs`, `read.rs`, `filters.rs`, `pairing.rs`, `extract.rs`, `reference.rs`,
`features.rs` (except depth: write 0), `output/*` for the core columns plus `patient_id`,
`sample_id`, `tumor_mut_present = 0`, `stats.rs`, a single-threaded `worker.rs`/`run.rs`.
Unit tests: CIGAR/MD walks (R-READ-2..5) including a proptest against a slow reference
implementation; HQ positions; eviction with stale FIFO names; entropy and hash vectors.
Integration test: S1 with `--ignore read_depth`.

Server test:
```bash
./target/release/bam2df <R1 arguments> --output_file golden/rust/R1.parquet --threads 1 --verbose > golden/rust/R1.log
./target/release/bam2df-compare parquet golden/R1/out.parquet golden/rust/R1.parquet --ignore read_depth
./target/release/bam2df-compare logs golden/R1/python.log golden/rust/R1.log
./target/release/bam2df-compare csv golden/R1/out_metadata.csv golden/rust/R1_metadata.csv
```
Pass: all three compare commands exit 0.

### Step 3: Read depth

Build: `depth.rs` per R-DEPTH and V1. Integration test: S1 in full.

Server test: as step 2 without `--ignore read_depth`. Pass: exit 0.

### Step 4: Blacklists in extraction

Build: region filters and germline sets wired into extraction (R-EXT). Integration test: S2.

Server test: R2 with the step-2 commands. Pass: exit 0.

### Step 5: Tracks, chromatin, labels, BED and match modes

Build: `tracks/bigwig.rs`, `tracks/chromatin.rs`, label lookup, the full schema and column order,
BED regions (R-IN-10, R-PAIR-1), match mode, single-read mode (R-PAIR-10), the reservoir (D11). Integration tests: S3, S4, S5, S8, S6
(logs only for S6, for `evicted=`).

Server test: R3, R4, R5 with the step-2 commands. Pass: all exit 0.

### Step 6: Parallel run, part files, performance

Build: R-RUN-2 scheduling, part files and row-group concat (R-OUT-5), scratch directory handling,
error propagation (R-RUN-3), timing lines.

Server test:
```bash
# same result on 1 and 12 threads
./target/release/bam2df <R3 arguments> --output_file golden/rust/R3_t12.parquet --threads 12
./target/release/bam2df-compare parquet golden/R3/out.parquet golden/rust/R3_t12.parquet
# full BAM, with the gwf job's resources
srun --account "$ACCOUNT" -c 12 --mem 50g --time 06:00:00 \
    /usr/bin/time -v ./target/release/bam2df <full arguments> --output_file golden/rust/full.parquet --verbose \
    > golden/rust/full.log 2> golden/rust/full.time
./target/release/bam2df-compare fingerprint golden/full/out.parquet golden/rust/full.parquet
./target/release/bam2df-compare logs golden/full/python.log golden/rust/full.log
./target/release/bam2df-compare csv golden/full/out_metadata.csv golden/rust/full_metadata.csv
```
Pass: all compares exit 0; record wall time and peak memory against Python's in the progress log;
lower the gwf memory request only after that.

### Step 7: gwf integration and rollout

Build: `make_dataset(..., engine="python")` in `src/gwf/template.py` with a `bam2df_bin` setting;
`engine="rust"` runs `{bam2df_bin} {args_block}` instead of `python ../src/bam2df.py`, with the
same `cores="12"`. In `src/gwf/workflow.py`, pass the mutation catalog as TSV (D5) when using Rust.
README: build instructions (`cargo build --release`), the Rust command line, the mutation catalog
format.

Server test: run a handful of BAMs through both engines with gwf; `bam2df-compare fingerprint` each
pair. Pass: all equal. Then switch the default engine to Rust. Removing the Python code is a
separate decision.

### Step 8 (optional): Deliberate fixes

One commit per item from section 6 or D4 widening, each with a note in this document of the
expected output change and a new golden run that shows only that change.

---

## 9. Scheduling at 12 cores

Longest-first placement of hg38 autosome lengths (Mb) onto 12 workers. The ideal is about 240 Mb per
worker; the longest worker is chr1 alone at 249 Mb, about 4% over, so per-chromosome jobs are
sufficient and tiles are not used.

| Worker | Chromosomes | Mb |
|---|---|---|
| 1 | chr1 | 249 |
| 2 | chr2 | 242 |
| 3 | chr3, chr21 | 245 |
| 4 | chr4, chr22 | 241 |
| 5 | chr5, chr19 | 240 |
| 6 | chr6, chr20 | 235 |
| 7 | chr7, chr18 | 240 |
| 8 | chr8, chr17 | 228 |
| 9 | chr9, chr16 | 229 |
| 10 | chr11, chr15 | 237 |
| 11 | chr10, chr14 | 241 |
| 12 | chr12, chr13 | 248 |

If measured per-chromosome times (the `Finished` log lines) differ much from length, sort by a table
of measured times instead.

---

## 10. Progress log

Update after each step: date, commit, result, notes.

- [ ] Step 0: golden fixtures
- [ ] Step 1: workspace, config, CLI, IDs, inputs, compare tool
- [ ] Step 2: core extraction
- [ ] Step 3: read depth
- [ ] Step 4: blacklists
- [ ] Step 5: tracks, chromatin, labels, BED and match modes
- [ ] Step 6: parallel run and performance
- [ ] Step 7: gwf integration and rollout
- [ ] Step 8: deliberate fixes (optional)
