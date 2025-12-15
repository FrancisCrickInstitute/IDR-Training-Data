import os
from pathlib import Path

dir2 = "Z:/working/barryd/IDR/crosstalk_training_data/source"
dir1 = "Z:/working/barryd/IDR/crosstalk_training_data/bleed"

# Extract base names (without _mixed or _source suffix)
mixed_files = {f.stem.replace('_mixed', '') for f in Path(dir1).glob('*_mixed.tif')}
source_files = {f.stem.replace('_source', '') for f in Path(dir2).glob('*_source.tif')}

# Find unmatched files
only_mixed = mixed_files - source_files
only_source = source_files - mixed_files

print(f"Files only in mixed directory: {len(only_mixed)}")
for f in sorted(only_mixed):
    print(f"  {f}_mixed.tif")

print(f"\nFiles only in source directory: {len(only_source)}")
for f in sorted(only_source):
    print(f"  {f}_source.tif")