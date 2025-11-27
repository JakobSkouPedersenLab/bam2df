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

## 2 .Install Poetry (inside the environment)
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

## Configuration

### JSON file

Extraction settings are defined in ```config.json```.
Example:
```json
{
  "bam": {
    "mm_rate_max": 1.0,
    "factor": 1.0,
    "context_flank": 5,
    "min_len": 0,
    "max_len": 299,
    "min_base_quality": 30,
    "min_mapq": 40,
    "max_nm": 2,
    "pre_sample_seed": 0,
    "n_match_sample": 0,
    "extract_type": "mismatch",
    "use_paired_reads": true
  }
}
```

### Key fields

- **extract_type**: "mismatch" for mismatch extraction
- **min_base_quality / min_mapq**:  read-level quality thresholds
- **max_nm**: maximum edit distance (mismathes) allowed
- **use_paired_reads**: whether to only sample in overlapping regions
- **n_match_sample**: sample a subset of matching bases (0 = all)

## Contributions

Contributions are welcome! If you find a bug or have an idea for a feature, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.