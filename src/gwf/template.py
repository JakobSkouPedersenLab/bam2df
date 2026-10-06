from gwf import AnonymousTarget

def make_dataset(bam_file_path,
                 reference_path,
                 config_path,
                 mutation_path=None,
                 genomic_blacklist_file=None,
                 donor_blacklist_file=None,
                 snp_blacklist_path=None,
                 lowpower_blacklist_path=None,
                 problematic_blacklist_path=None,
                 bed_intervals_path=None,
                 chromatin_path=None,
                 replication_path=None,
                 mappability_path=None,
                 microsatellite_path=None,
                 repeatmasker_path=None,
                 sample_id=None,
                 patient_id=None,
                 account=None,
                 output_file=None):
    """
    Run the per-BAM extraction pipeline and write a single parquet file.
    """
    inputs = [
        bam_file_path,
        reference_path,
        config_path,
    ]

    optional_paths = {
        "--mutation_path": mutation_path,
        "--genomic_blacklist_file": genomic_blacklist_file,
        "--donor_blacklist_file": donor_blacklist_file,
        "--snp_blacklist_path": snp_blacklist_path,
        "--lowpower_blacklist_path": lowpower_blacklist_path,
        "--problematic_blacklist_path": problematic_blacklist_path,
        "--bed_intervals_path": bed_intervals_path,
        "--chromatin_path": chromatin_path,
        "--replication_path": replication_path,
        "--mappability_path": mappability_path,
        "--microsatellite_path": microsatellite_path,
        "--repeatmasker_path": repeatmasker_path,
    }

    # Add each optional file only if present
    for f in optional_paths.values():
        if f:
            inputs.append(f)

    optional_args = dict(optional_paths)
    optional_args["--sample_id"] = sample_id
    optional_args["--patient_id"] = patient_id

    outputs = [output_file]
    options = dict(cores="12", memory="50g", walltime="06:00:00")

    # Cluster account, where the backend needs one. The caller supplies it; none is hardcoded.
    if account:
        options["account"] = account

    # Only pass arguments that are set: an empty one would render as a bare flag and
    # argparse would take the following flag to be its value.
    argv = [bam_file_path, "$TEMP/ref.fa", config_path]
    argv += [f"{flag} {value}" for flag, value in optional_args.items() if value]
    argv += [f"--output_file {output_file}", "--verbose"]
    args_block = " \\\n        ".join(argv)

    spec = f"""
    set -euo pipefail

    mkdir -p $TEMP/

    # Copy reference to scratch
    cp {reference_path} $TEMP/ref.fa
    cp {reference_path}.fai $TEMP/ref.fa.fai

    python ../src/bam2df.py \\
        {args_block}

    rm -f $TEMP/ref.fa $TEMP/ref.fa.fai
    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)
