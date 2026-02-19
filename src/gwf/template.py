from gwf import AnonymousTarget

def make_dataset(bam_file_path, reference_path, config_path, mutation_path=None,
                 genomic_blacklist_file=None, donor_blacklist_file=None,
                 snp_blacklist_path=None, lowpower_blacklist_path=None,
                 problematic_blacklist_path=None, chromatin_path=None,
                 replication_path=None, output_file=None):
    """
    Run the per-BAM extraction pipeline and write a single parquet file.
    """
    inputs = [
        bam_file_path,
        reference_path,
        config_path,
    ]
    
    # Add each optional file only if present
    optional_files = [
        mutation_path,
        genomic_blacklist_file,
        donor_blacklist_file,
        snp_blacklist_path,
        lowpower_blacklist_path,
        problematic_blacklist_path,
        chromatin_path,
        replication_path
    ]
    
    for f in optional_files:
        if f:
            inputs.append(f)
    
    outputs = [output_file]
    options = dict(cores="12", memory="50g", walltime="06:00:00", account="c2i-colon")

    spec = f"""
    set -euo pipefail

    mkdir -p $TEMP/
    
    # Copy reference to scratch
    cp {reference_path} $TEMP/ref.fa
    cp {reference_path}.fai $TEMP/ref.fa.fai
    
    python ../src/bam2df.py \
        {bam_file_path} \
        $TEMP/ref.fa \
        {config_path} \
        --mutation_path {mutation_path} \
        --genomic_blacklist_file {genomic_blacklist_file} \
        --donor_blacklist_file {donor_blacklist_file} \
        --snp_blacklist_path {snp_blacklist_path} \
        --lowpower_blacklist_path {lowpower_blacklist_path} \
        --problematic_blacklist_path {problematic_blacklist_path} \
        --chromatin_path {chromatin_path} \
        --replication_path {replication_path} \
        --output_file {output_file} \
        --verbose
        
    rm -f $TEMP/ref.fa $TEMP/ref.fa.fai
    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)
