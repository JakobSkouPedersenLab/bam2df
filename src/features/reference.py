from typing import Dict, Tuple
import pysam

coverage_cache: dict[tuple[str, int], list[int]] = ({})  # Key: (contig, segment_start); Value: coverage array for that segment
CACHE_SEGMENT_SIZE = 10_000 # e.g., cache coverage in 1000-base segments to reuse across close positions
MAX_SEGMENTS_CACHED = 20  # Limit cache size to avoid memory blow-up in extremely deep coverage scenarios
current_cache_contig = None  # Track current contig in cache for easy invalidation
bamfile_cache = {}  # Cache for opened BAM files

class ReferenceCache:
    """Fetch reference chunks and keep a sliding window cache per contig."""

    def __init__(self, fasta_path: str, window: int = 1_000_000):
        self.fasta = pysam.FastaFile(fasta_path)
        self.window = window
        self._cache: Dict[str, Tuple[int, int, str]] = {}

    def fetch(self, contig: str, pos: int, flank: int) -> str:
        """Return *2 × flank + 1* bases centred on 1-based **pos**."""
        start = max(pos - flank - 1, 0)
        end = pos + flank
        lo, hi, seq = self._cache.get(contig, (None, None, None))
        if seq and lo is not None and hi is not None and lo <= start and end <= hi:
            off = start - lo
            return seq[off : off + (2 * flank + 1)]
        new_lo = max(0, pos - self.window // 2)
        new_hi = new_lo + self.window
        seq = self.fasta.fetch(contig, new_lo, new_hi)
        self._cache[contig] = (new_lo, new_hi, seq)
        off = start - new_lo
        return seq[off : off + (2 * flank + 1)]