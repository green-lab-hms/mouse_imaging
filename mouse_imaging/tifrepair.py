"""
Workaround for recordings whose last TIFF was cut off, usually because MATLAB/ScanImage crashed during the
recording: the file ends partway through its last frame, so neither ScanImageTiffReader nor tifffile (and therefore
suite2p) can read it.

`prepare_tif_list` returns the list of TIFFs suite2p should read. If the last one is cut off, it writes a repaired copy
(in the session's derived folder) that keeps every complete volume of that file and drops the incomplete rest. The raw
file is not changed. The copy is made by truncating the original bytes at the start of the first dropped page and
ending the page list there, so ScanImage's own header and format are kept exactly.
"""
import os, json, struct, time, shutil
from pathlib import Path

def _pages(tif):
    """(IFD offset, data end) of every page tifffile can list, plus file layout info."""
    import tifffile
    with tifffile.TiffFile(tif) as t:
        pages = [(p.offset, max(o + n for o, n in zip(p.dataoffsets, p.databytecounts))) for p in t.pages]
        return pages, t.is_bigtiff, t.byteorder

def n_frames(tif):
    import tifffile
    with tifffile.TiffFile(tif) as t:
        return len(t.pages)

def truncation(tif):
    """None if the last page's data is complete, else (complete pages, listed pages)."""
    pages, _, _ = _pages(tif)
    size = os.path.getsize(tif)
    complete = sum(end <= size for _, end in pages)
    return None if complete == len(pages) else (complete, len(pages))

def write_truncated_copy(src, dst, keep):
    """Copy the first `keep` pages of src to dst, byte for byte, and end the page list after them."""
    pages, big, byteorder = _pages(src)
    assert 0 < keep < len(pages), (keep, len(pages))
    cut = pages[keep][0]  # IFD of the first dropped page; everything before it belongs to the kept pages
    assert pages[keep - 1][1] <= cut, 'unexpected page layout (data after the next page directory)'
    tmp = Path(str(dst) + '.tmp')
    with open(src, 'rb') as fin, open(tmp, 'wb') as fout:
        remaining = cut
        while remaining:
            chunk = fin.read(min(remaining, 64 << 20))
            fout.write(chunk)
            remaining -= len(chunk)
        # Point the last kept page's "next page" offset at nothing
        last_ifd = pages[keep - 1][0]
        count_fmt, entry, off_fmt = ('Q', 20, 'Q') if big else ('H', 12, 'I')
        fin.seek(last_ifd)
        ntags = struct.unpack(byteorder + count_fmt, fin.read(struct.calcsize(count_fmt)))[0]
        fout.seek(last_ifd + struct.calcsize(count_fmt) + ntags * entry)
        fout.write(struct.pack(byteorder + off_fmt, 0))
    tmp.replace(dst)

def prepare_tif_list(raw_dir, out_dir, frames_per_volume):
    """
    Absolute paths of the TIFFs in raw_dir for suite2p. If the last one is cut off, it is replaced by a repaired copy in
    out_dir that ends at the last complete volume, and out_dir/repair.json records what was dropped.
    Returns (paths, repair info or None).
    """
    tifs = sorted(Path(raw_dir).glob('*.tif'))
    if not tifs:
        return [], None
    cut = truncation(tifs[-1])
    if cut is None:
        return [str(t) for t in tifs], None
    complete, listed = cut
    before = sum(n_frames(t) for t in tifs[:-1])
    total = before + complete
    keep = complete - total % frames_per_volume  # whole volumes only, so every plane and channel stays aligned
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repaired = out_dir / tifs[-1].name
    write_truncated_copy(tifs[-1], repaired, keep)
    info = {'file': str(tifs[-1]), 'repaired_copy': str(repaired), 'pages_listed': listed, 'pages_complete': complete,
            'pages_kept': keep, 'frames_dropped': listed - keep, 'frames_per_volume': frames_per_volume,
            'volumes_kept': (before + keep) // frames_per_volume, 'created': time.strftime('%Y-%m-%d %H:%M')}
    (out_dir / 'repair.json').write_text(json.dumps(info, indent=1))
    print(f"Last TIFF {tifs[-1].name} is cut off ({complete} of {listed} pages complete); suite2p reads a repaired copy "
          f"with {keep} pages ({info['volumes_kept']} whole volumes in all, {listed - keep} frames dropped).")
    return [str(t) for t in tifs[:-1]] + [str(repaired)], info

def remove_copies(out_dir):
    """Delete repaired copies after suite2p has read them (repair.json is kept as the record)."""
    for f in Path(out_dir).glob('*.tif'):
        f.unlink()
