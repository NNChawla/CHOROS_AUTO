"""
Pack a directory of per-session .npy files into a single memory-mapped array.

Eliminates per-sample file-open overhead in VRDataset.__getitem__ by replacing
71K individual file seeks with reads from one contiguous mmap.  The OS page
cache caches hot regions, and multiple training processes on the same machine
share the same physical pages automatically.

Output (written to npy_dir's parent directory):
  <name>_packed.npy   — float32 C-order array, shape (total_rows, n_cols)
  <name>_index.npz    — names (N,) str, offsets (N,) int64, n_rows (N,) int64

Usage:
  conda run -n CHOROS python setup/pack_dataset.py /srv/CHOROS/data/kinematics/VR_npy_PVAJ

Rerunning is safe — existing output files are overwritten.
"""

import argparse
import ast
import struct
import numpy as np
from pathlib import Path


def _read_npy_header(path: Path):
    with open(path, 'rb') as f:
        f.read(6)
        major = struct.unpack('B', f.read(1))[0]
        f.read(1)
        hlen = struct.unpack('<H' if major == 1 else '<I',
                             f.read(2 if major == 1 else 4))[0]
        hdr = ast.literal_eval(f.read(hlen).decode('latin1').strip().rstrip(','))
        data_offset = f.tell()
    shape = hdr['shape']
    dtype = np.dtype(hdr['descr'])
    fortran = hdr.get('fortran_order', False)
    return data_offset, shape, dtype, fortran


def main():
    parser = argparse.ArgumentParser(description='Pack npy dir into single mmap file')
    parser.add_argument('npy_dir', help='Directory containing .npy session files')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: parent of npy_dir)')
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else npy_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    packed_path = out_dir / f'{npy_dir.name}_packed.npy'
    index_path  = out_dir / f'{npy_dir.name}_index.npz'

    files = sorted(npy_dir.glob('*.npy'))
    if not files:
        raise RuntimeError(f'No .npy files found in {npy_dir}')
    print(f'Found {len(files):,} .npy files in {npy_dir}')

    print('Reading headers …')
    headers = [_read_npy_header(f) for f in files]

    n_cols_set = {h[1][1] for h in headers}
    if len(n_cols_set) != 1:
        raise RuntimeError(f'Inconsistent n_cols across files: {n_cols_set}')
    n_cols = n_cols_set.pop()
    total_rows = sum(h[1][0] for h in headers)
    size_gb = total_rows * n_cols * 4 / 1e9
    print(f'Total: {total_rows:,} rows × {n_cols} cols  →  {size_gb:.2f} GB')
    print(f'Writing {packed_path} …')

    out = np.lib.format.open_memmap(packed_path, mode='w+', dtype='float32', shape=(total_rows, n_cols))

    names, offsets, n_rows_arr = [], [], []
    row = 0
    for i, (f, (data_offset, shape, dtype, fortran)) in enumerate(zip(files, headers)):
        if i % 5000 == 0:
            print(f'  {i:,}/{len(files):,}  ({100*i/len(files):.1f}%)  row {row:,}', flush=True)
        nr = shape[0]
        if fortran:
            data = np.array(np.load(f, mmap_mode='r'), dtype=np.float32)
        else:
            with open(f, 'rb') as fh:
                fh.seek(data_offset)
                raw = np.frombuffer(fh.read(nr * n_cols * dtype.itemsize), dtype=dtype)
            data = raw.reshape(nr, n_cols).astype(np.float32)
        out[row : row + nr] = data
        names.append(f.name)
        offsets.append(row)
        n_rows_arr.append(nr)
        row += nr

    out.flush()
    del out
    print(f'Flushed {packed_path}')

    np.savez(index_path,
             names=np.array(names),
             offsets=np.array(offsets, dtype=np.int64),
             n_rows=np.array(n_rows_arr, dtype=np.int64))
    print(f'Saved index → {index_path}')
    print('Done.')


if __name__ == '__main__':
    main()
