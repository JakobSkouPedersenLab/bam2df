# bam2df
A lightweight command-line tool that converts BAM files into model-ready feature tables.  
It extracts per-base or per-mismatch features, applies blacklist filters, integrates mutation and coverage data, and outputs:

- `<sample_id>.parquet` — main feature table  
- `<sample_id>_metadata.csv` — metadata for the processed BAM  

All output paths are configurable, and the tool is optimized for large-scale genomics pipelines.

---

## Setting up your environment

### 1. Create a new Conda environment
```bash
conda create -n bam2df python=3.12
conda activate bam2df
```

## 2 Install Poetry (inside the environment)
```bash
pip install poetry
poetry install
```

## How to use

### Command line

Once the environment is activated, run the main script:
```bash
python ./src/bam2df.py \
    /path/to/sample.bam \
    /path/to/reference.fasta \
    ./config.json \
    --mutation_path ./data/interim/mutations/<sample_id>.pkl \
    --genomic_blacklist_file /path/to/genomic_blacklist.bed \
    --donor_blacklist_file /path/to/donor_recurrent.tsv \
    --snp_blacklist_path /path/to/snps.tsv \
    --lowpower_blacklist_path /path/to/lowpower.bg \
    --problematic_blacklist_path /path/to/problematic_folder/ \
    --coverage_path /path/to/coverage.bw \
    --chromatin_path /path/to/chromatin_features.bw \
    --replication_path /path/to/replication_timing.bw \
    --output_dir ./data \
    --verbose
```

### Arguments Explained

- **bam_path**: Input BAM file.
- **reference_path**: Reference FASTA file.
- **config_path**: JSON configuration specifying extraction settings.
- **--mutation_path** *(optional)*: Mutation index used for labeling mismatches.
- **--genomic_blacklist_file** *(optional)*: BED file listing genomic regions that should be excluded.
- **--donor_blacklist_file** *(optional)*: Donor-specific blacklist (TSV).
- **--snp_blacklist_path** *(optional)*: SNP calls to exclude.
- **--lowpower_blacklist_path** *(optional)*: Regions with insufficient coverage.
- **--problematic_blacklist_path** *(optional)*: Directory containing additional region filters.
- **--coverage_path** *(optional)*: bigWig file with per-base coverage.
- **--chromatin_path** *(optional)*: Chromatin accessibility track (eg, bigWig).
- **--replication_path** *(optional)*: Replication timing track (eg, bigWig).
- **--output_dir**: Output directory for generated data (default: `./data`).
- **--verbose**: Enables detailed logging.

### Output

By default, outputs are written to:
```bash
./data/
```

with the following structure:
```bash
data/
  <sample_id>.parquet
  <sample_id>_metadata.csv
```

You can customize the output location via:
```bash
--output_dir /desired/output/path
```

### Extracted Features

Each row in the output parquet file corresponds to a single read position and contains the following features:

| Field | Description |
|-------|-------------|
| qname | Read name (query name) |
| chr | Chromosome |
| genomic_pos | genomic coordinate |
| obs_type | Observation type (eg, `S` for `SNV`) |
| obs_base | Observed base(s) in read |
| obs_ref | Reference base(s) |
| obs_alt_len | Length of alternate allele |
| obs_ref_len | Length of reference allele |
| len_delta | Length difference (`obs_alt_len - obs_ref_len`) |
| strand | Read strand (`+` or `-`) |
| first_in_pair | 1 if first read in pair, 0 otherwise |
| read_index | Base index within the read |
| fragment_len | Inferred fragment length |
| seq_length | Total read length |
| trinucleotide_ctx | 3-bp reference context centered on position |
| start_ctx | 5-bp reference context at read start |
| end_ctx | 5-bp reference context at read end |
| reference_ctx | 5-bp reference context centered on position |
| local_gc | Local GC fraction in surrounding window |
| local_complexity_1 | Local sequence complexity (metric 1) |
| local_complexity_2 | Local sequence complexity (metric 2) |
| n_other_err | Number of other alignment errors in read |
| base_qual_site | Base quality at position (Phred) |
| dist_to_end | Distance to nearest read end (bp) |
| homopolymer_run_len | Length of homopolymer run at position |
| read_depth | Read depth (coverage) at genomic position |
| replication_timing | Replication timing value at genomic position |

## Configuration

### JSON file

Extraction settings are defined in ```config.json```.
Example:
```json
{
  "bam": {
    "context_flank": 5,
    "min_len": 0,
    "max_len": 299,
    "min_base_quality": 30,
    "min_mapq": 40,
    "max_nm": 2,
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

- **context_flank**: Number of flanking bases to include on each side of the variant position.
- **min_len / max_len**: Minimum and maximum read length allowed.
- **min_base_quality**: Minimum base quality (Phred) required at the variant position.
- **min_mapq**: Minimum read mapping quality required.
- **max_nm**: Maximum edit distance (number of mismatches/indels) allowed per read.
- **bin_size_bp**: Bin size (in base pairs) used for genomic binning or aggregation.
- **pre_sample_seed**: Random seed used for deterministic subsampling.
- **n_match_sample**: Number of matching (reference) bases to subsample (0 = keep all).
- **mutation_types**: Which mutation classes to extract (eg, `mismatch`, `indel`).
- **use_paired_reads**: If true, only sample bases from overlapping regions of paired-end reads.
- **chunk_size**: Number of reads processed per chunk for streaming and memory control.

## Contributions

Contributions are welcome! If you find a bug or have an idea for a feature, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
