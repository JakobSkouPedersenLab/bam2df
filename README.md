# bam2df

A lightweight command-line tool that converts BAM files into model-ready feature tables.

`bam2df` extracts read-level observations from BAM files, including mismatches, indels, and optionally sampled matching bases. It supports blacklist filtering, BED interval restriction, mutation-label annotation, and multiple genomic context feature tracks.

The tool outputs:

- `<sample_id>.parquet` — main feature table
- `<sample_id>_metadata.csv` — metadata for the processed BAM file

The pipeline is designed for large BAM files and writes intermediate chunks before producing a single compressed parquet output.

---

## Setting up your environment

### 1. Create a new Conda environment

```bash
conda create -n bam2df python=3.12
conda activate bam2df
```

### 2. Install Poetry

```bash
pip install poetry
poetry install
```

### 3. Install gwf, if needed

```bash
conda config --add channels gwforg
conda install gwf
```

---

## How to use

### Command line

```bash
python ./src/bam2df.py \
    /path/to/sample.bam \
    /path/to/reference.fasta \
    ./config.json \
    --output_file /path/to/output/sample.parquet \
    --mutation_path /path/to/mutations/sample.pkl \
    --genomic_blacklist_file /path/to/genomic_blacklist.bed \
    --donor_blacklist_file /path/to/donor_blacklist.tsv \
    --snp_blacklist_path /path/to/snps.tsv \
    --lowpower_blacklist_path /path/to/lowpower.bg \
    --problematic_blacklist_path /path/to/problematic.bg \
    --bed_intervals_path /path/to/target_intervals.bed \
    --chromatin_path /path/to/chromatin_features.bw \
    --replication_path /path/to/replication_timing.bw \
    --mappability_path /path/to/mappability.bw \
    --microsatellite_path /path/to/microsatellites.bed \
    --repeatmasker_path /path/to/repeatmasker.bed \
    --verbose
```

Only the three positional arguments are required by `argparse`:

```bash
python ./src/bam2df.py \
    /path/to/sample.bam \
    /path/to/reference.fasta \
    ./config.json \
    --output_file /path/to/output/sample.parquet
```

Note: `--output_file` is currently required in practice because the script calls `Path(args.output_file).resolve()`.

---

## Arguments

### Required positional arguments

| Argument | Description |
|---|---|
| `bam_path` | Input BAM file. |
| `reference_path` | Reference FASTA file. |
| `config_path` | JSON configuration file specifying extraction settings. |

### Output and logging

| Argument | Description |
|---|---|
| `--output_file` | Output parquet file. Parent directories are created automatically. |
| `--verbose` | Enable detailed logging. |

### Mutation labeling

| Argument | Description |
|---|---|
| `--mutation_path` | Optional pickle file containing mutation records used to label extracted observations. |

### Blacklist filters

| Argument | Description |
|---|---|
| `--genomic_blacklist_file` | BED file with genomic regions to exclude. |
| `--donor_blacklist_file` | Donor-level blacklist table. |
| `--snp_blacklist_path` | Patient germline variant calls. Observations matching a call's exact allele are excluded (see below). |
| `--lowpower_blacklist_path` | Low-power or low-coverage regions to exclude. |
| `--problematic_blacklist_path` | Problematic regions to exclude. |

#### SNP blacklist

The SNP blacklist removes the patient's own germline variants, matched on **exact allele**: an
SNV observation is dropped only if its chromosome, position, REF and ALT all match a call, and
an indel only if its anchored REF and ALT match. A different base at a germline site, which is a
sequencing error rather than the germline allele, is still extracted.

The file is tab-separated with a header containing at least `CHR`, `START`, `REF` and `ALT`
(case-insensitive, an optional leading `#`, e.g. `#CHR`):

- `START` is **0-based**. `END`, if present, is not used.
- `REF`/`ALT` are VCF-style; indels are anchored on the preceding reference base
  (insertion `C → CTT`, deletion `GAA → G`). `ALT` may list several alleles, comma-separated.
- Multi-base substitutions, complex indels and `*`/`.` alleles can never match a bam2df
  observation; they are counted in the log and skipped.

At startup, up to 10,000 SNV REF alleles are checked against the reference FASTA. If more than
5% disagree, which is what a 1-based `START` or a different assembly produces, the run fails.
A missing SNP blacklist file is an error.

### BED interval restriction

| Argument | Description |
|---|---|
| `--bed_intervals_path` | Optional BED file specifying genomic intervals to process. If provided, extraction is restricted to these intervals. |

### Optional genomic context features

| Argument | Description |
|---|---|
| `--chromatin_path` | Chromatin accessibility feature track, for example ATAC/DNase-derived features. |
| `--replication_path` | Replication timing feature track. |
| `--mappability_path` | Mappability feature track. |
| `--microsatellite_path` | Microsatellite annotation track. |
| `--repeatmasker_path` | RepeatMasker annotation track. |

---

## Output

For an output path such as:

```bash
--output_file /path/to/output/C05638.parquet
```

the pipeline writes:

```text
/path/to/output/
  C05638.parquet
  C05638_metadata.csv
```

The parquet file contains the extracted read-level observations and optional context features. The metadata CSV contains per-file extraction statistics and is written beside the parquet file.

The output table also includes:

| Field | Description |
|---|---|
| `patient_id` | Patient ID inferred from the BAM path. |
| `sample_id` | Sample ID inferred from the BAM path. |
| `tumor_label` | Label indicating whether the observation matches the supplied mutation catalog. |
| `somatic_label` | Somatic mutation label, if available from the mutation labeling step. |

---

## Extracted features

Each row in the output parquet file corresponds to an extracted read-level observation.

Core columns may include:

| Field | Description |
|---|---|
| `qname` | Read name/query name. |
| `chr` | Chromosome. |
| `genomic_pos` | Genomic coordinate. |
| `obs_type` | Observation type, for example mismatch, insertion, deletion, or match. |
| `obs_alt` | Observed alternate allele or base in the read. |
| `obs_ref` | Reference allele or base. |
| `obs_alt_len` | Length of observed alternate allele. |
| `obs_ref_len` | Length of reference allele. |
| `len_delta` | Length difference, `obs_alt_len - obs_ref_len`. |
| `strand` | Read strand, `+` or `-`. |
| `first_in_pair` | Whether the observation comes from the first read in a pair. |
| `read_index` | Base index within the read. |
| `fragment_len` | Inferred fragment length. |
| `seq_length` | Read sequence length. |
| `trinucleotide_ctx` | 3-bp reference context centered on the observation. |
| `start_ctx` | Query context near the fragment start. |
| `end_ctx` | Query context near the fragment end. |
| `reference_ctx` | Reference context centered on the observation. |
| `local_gc` | Local GC fraction in the surrounding sequence window. |
| `local_complexity_1` | Local sequence complexity feature. |
| `local_complexity_2` | Additional local sequence complexity feature. |
| `n_other_err` | Number of other alignment errors in the read/read pair. |
| `base_qual_site` | Base quality at the observation position. |
| `dist_to_end` | Distance to nearest read end. |
| `homopolymer_run_len` | Homopolymer run length at the observation position. |
| `read_depth` | Read depth at the genomic position. |

Depending on which optional feature paths are provided, additional columns may be added for:

- replication timing
- mappability
- chromatin accessibility
- microsatellite annotations
- repeat annotations from RepeatMasker

---

## BED interval mode

The CLI supports restricting extraction to a set of genomic intervals:

```bash
--bed_intervals_path /path/to/intervals.bed
```

When this option is provided, `bam2df` only processes observations within the supplied BED intervals. This is useful for targeted experiments, debugging, panel-style extraction, or reducing runtime when only specific genomic regions are needed.

If no BED intervals are provided, extraction is performed genome-wide subject to the configured filters and blacklists.

---

## Using a mutation catalog for labeling

The pipeline optionally supports labeling extracted observations using an external mutation catalog.

```bash
--mutation_path /path/to/mutations.pkl
```

If a mutation catalog is provided, the pipeline compares each extracted observation against known mutations and adds mutation labels to the output table.

If no mutation catalog is provided, the pipeline still runs and continues without mutation-index labeling.

### Expected mutation catalog format

The mutation catalog should be a Python pickle file containing mutation records compatible with:

```python
src.features.labeling.add_tumor_mutation_labels
```

A typical record contains:

```text
(chr, genomic_pos, ref, alt)
```

See the relevant preprocessing utilities in `src/utility/io.py` for how mutation indexes are created.

---

## Configuration

Extraction settings are defined in `config.json`.

Example:

```json
{
  "bam": {
    "context_flank": 5,
    "min_len": 0,
    "max_len": 299,
    "min_base_quality": 30,
    "min_mapq": 40,
    "bin_size_bp": 10,
    "pre_sample_seed": 0,
    "n_match_sample": 0,
    "mutation_types": ["mismatch", "indel"],
    "use_paired_reads": true,
    "chunk_size": 50000
  }
}
```

### Key fields

| Field | Description |
|---|---|
| `context_flank` | Number of flanking bases to include around the observation. |
| `min_len` / `max_len` | Minimum and maximum read or fragment length allowed. |
| `min_base_quality` | Required. A position is scanned only where both mates reach this base quality. The former key `min_bq` is rejected. |
| `min_mapq` | Minimum mapping quality required. |
| `bin_size_bp` | Bin size in base pairs used for genomic binning or aggregation. |
| `pre_sample_seed` | Random seed used for deterministic subsampling. |
| `n_match_sample` | Number of matching/reference bases to subsample. Use `0` to keep all configured observations. |
| `mutation_types` | Mutation or observation classes to extract, for example `mismatch` and `indel`. |
| `use_paired_reads` | If true, use paired-read information during extraction. |
| `chunk_size` | Number of reads processed per chunk for streaming and memory control. |

---

## Parallel execution

The number of workers is read from the environment variable:

```bash
SLURM_CPUS_PER_TASK
```

For example:

```bash
SLURM_CPUS_PER_TASK=8 python ./src/bam2df.py \
    /path/to/sample.bam \
    /path/to/reference.fasta \
    ./config.json \
    --output_file /path/to/output/sample.parquet
```

If `SLURM_CPUS_PER_TASK` is not set, the pipeline defaults to one worker.

---

## Tests

```bash
poetry install --with dev
poetry run pytest
```

---

## Contributions

Contributions are welcome. If you find a bug or have an idea for a feature, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
