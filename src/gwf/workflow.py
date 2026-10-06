#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List
from gwf import Workflow

from src.gwf.template import make_dataset
from src.utility.sample import get_patient_id, get_sample_id

def read_text_file(path: str) -> List[str]:
    """
    Reads a text file
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
    
def resolve_patient_dir(base: Path, patient_id: str) -> Path | None:
    """
    Returns the actual on-disk patient directory under `base`,
    matching case-insensitively.
    """
    if not base.exists():
        return None

    pid_lower = patient_id.lower()

    for child in base.iterdir():
        if child.is_dir() and child.name.lower() == pid_lower:
            return child

    return None

gwf = Workflow()

config_path = "../../config.json"

bam_file_list_path = "/path/to/bam_paths.txt"  # text file listing one BAM path per line

# Site-specific settings. Fill these in before running; they are placeholders so that no
# real paths or cluster account names end up in version control.
slurm_account = ""
reference_file = "/path/to/reference.fasta"
genomic_blacklist_file = "/path/to/genomic_blacklist.bed"
donor_blacklist_file = "/path/to/donor_blacklist.tsv"
snp_blacklist_path = "/path/to/snp_blacklists"
lowpower_blacklist_path = "/path/to/lowpower_blacklists"
problematic_blacklist_path = "/path/to/problematic_blacklists"
bed_intervals_path = "/path/to/bed_intervals"
chromatin_path = "/path/to/chromatin"
replication_path = "/path/to/replication_timing"
mappability_path = "/path/to/mappability"
microsatellite_path = "/path/to/microsatellite"
repeatmasker_path = "/path/to/repeatmasker"

bam_paths = read_text_file(bam_file_list_path)

processed_samples: set[str] = set()

# Prepare output directory once
output_dir = Path("../../data")
output_dir.mkdir(parents=True, exist_ok=True)

# Process BAM files
for bam_path in bam_paths:
    
    patient_id = get_patient_id(bam_path)
    sample_id = get_sample_id(bam_path)
    if sample_id in processed_samples:
        continue
    processed_samples.add(sample_id)
    
    # Set output file
    output_file = output_dir / f"{sample_id}.parquet"
    
    # Mutation catalog for this patient, if one is available. Substitute `patient_id`
    # into the path here if the catalogs are stored per patient.
    mutation_path = Path("/path_to_patient_mutation_catalog.pkl")
    mutation_file = str(mutation_path) if mutation_path.exists() else None
                
    # Resolve patient-specific SNP blacklist
    patient_snp = None
    if snp_blacklist_path:
        base = Path(snp_blacklist_path)
        patient_dir = resolve_patient_dir(base, patient_id)

        if patient_dir:
            snp_candidate = patient_dir / f"{patient_dir.name}_snpcalls_QUAL20_stack_geQbinom.tsv"
            if snp_candidate.exists():
                patient_snp = str(snp_candidate)
            else:
                print(f"[WARN] No SNP blacklist for {patient_id}: {snp_candidate}", flush=True)
        else:
            print(f"[WARN] No SNP directory for {patient_id} under {base}", flush=True)
    
    # Resolve patient-specific lowpower blacklist
    patient_lowpower = None
    if lowpower_blacklist_path: 
        base = Path(lowpower_blacklist_path)
        patient_dir = resolve_patient_dir(base, patient_id)

        if patient_dir:
            lp_candidate = patient_dir / f"{patient_dir.name}.lowpower.lt100X.bg"
            if lp_candidate.exists():
                patient_lowpower = str(lp_candidate)
            else:
                print(f"[WARN] No lowpower blacklist for {patient_id}: {lp_candidate}", flush=True)
        else:
            print(f"[WARN] No lowpower directory for {patient_id} under {base}", flush=True)
    
    # Resolve patient-specific problematic blacklist
    patient_problematic = None
    if problematic_blacklist_path:
        base = Path(problematic_blacklist_path)
        patient_dir = resolve_patient_dir(base, patient_id)

        if patient_dir:
            matches = sorted(patient_dir.glob(f"{patient_dir.name}.problematic.merged.gt*X.bg"))

            if len(matches) == 1:
                patient_problematic = str(matches[0])
            elif len(matches) > 1:
                patient_problematic = str(matches[0])
                print(
                    f"[WARN] Multiple problematic blacklist files for {patient_id}, using first: {matches}",
                    flush=True,
                )
            else:
                print(
                    f"[WARN] No problematic blacklist for {patient_id} in {patient_dir}",
                    flush=True,
                )
        else:
            print(f"[WARN] No problematic blacklist directory for {patient_id} under {base}", flush=True)
    
    # Pass everything to the template
    template_args = {
        "bam_file_path": bam_path,
        "mutation_path": mutation_file,
        "reference_path": reference_file,
        "config_path": config_path,
        "genomic_blacklist_file": genomic_blacklist_file,
        "donor_blacklist_file": donor_blacklist_file,
        "snp_blacklist_path": patient_snp,
        "lowpower_blacklist_path": patient_lowpower,
        "problematic_blacklist_path": patient_problematic,
        "bed_intervals_path": bed_intervals_path,
        "chromatin_path": chromatin_path,
        "replication_path": replication_path,
        "mappability_path": mappability_path,
        "microsatellite_path": microsatellite_path,
        "repeatmasker_path": repeatmasker_path,
        "account": slurm_account,
        "output_file": str(output_file)
    }

    gwf.target_from_template(
        name=f"make_raw_dataset_{sample_id}",
        template=make_dataset(**template_args),
    ) 
