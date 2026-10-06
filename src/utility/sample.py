import os
import re

def get_sample_id(path: str) -> str:
    """
    Extract sample ID from BAM or VCF path.
    
    Supports patterns like:
      - C00001/C00A00001D
      - C00001/C00T00001D
      - C00002/C00M00002D
      - donor### (case-insensitive → normalized to Donor###)
    """
    filename = os.path.basename(path)

    # Case 1: combined patient_sample pattern e.g. C00001_C00A00001D
    m0 = re.search(r"(C\d{5})_+(C\d+[A-Z]\d{5}D?)", filename)
    if m0:
        return m0.group(2)

    # Case 2: Donor pattern (case-insensitive)
    m_donor = re.search(r"(donor\d+)", path, re.IGNORECASE)
    if m_donor:
        donor_id = m_donor.group(1)
        return donor_id[0].upper() + donor_id[1:]

    # Case 3: Generic sample pattern anywhere — allows any uppercase letter variant
    m_sample = re.search(r"(C\d+[A-Z]\d{5}D?)", path)
    if m_sample:
        return m_sample.group(1)

    raise ValueError(f"Could not find sample ID in: {path}")

def get_patient_id(path: str) -> str:
    """Extract patient ID from BAM or VCF path."""
    sample_id = get_sample_id(path)
    filename = os.path.basename(path)

    # Case 1: "C#####_C##X#####D" pattern
    m0 = re.search(r"(C\d{5})_+(C\d+[A-Z]\d{5}D?)", filename)
    if m0:
        return m0.group(1)

    # Case 2: Donor pattern (case-insensitive)
    m_donor = re.search(r"(donor\d+)", path, re.IGNORECASE)
    if m_donor:
        donor_id = m_donor.group(1)
        return donor_id[0].upper() + donor_id[1:]

    # Case 3: Patient ID in directory structure
    m_dir = re.search(r"/(C\d{5})(?:/|$)", path)
    if m_dir:
        return m_dir.group(1)

    # Case 4: Derive from sample ID digits after letter (A/T/M/etc.)
    m_pat = re.search(r"[A-Z](\d{5})", sample_id)
    if m_pat:
        return f"C{m_pat.group(1)}"

    return sample_id