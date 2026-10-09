"""
Lossless compression of a session's raw ScanImage TIFFs, run as the last preprocessing step (step 'compress').

Each <name>.tif becomes <name>.tif.zst: the whole file compressed with zstd (level 6, about 57% of the original size
for our recordings). Decompressing gives back the identical file, ScanImage header included, and any zstd tool can do
it: `zstd -d <name>.tif.zst`. A file is only replaced after its .zst has been decompressed and checked against the
original's SHA-256.

Compressed TIFFs live in the session folder and its subfolders (filter*, ref, ...). Steps that read raw TIFFs
(suite2p, filters) decompress them first (`decompress_dir`), and the compress step compresses them again afterwards.
"""
import os, hashlib, tempfile, contextlib, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SUFFIX = '.zst'
LEVEL = 6
CHUNK = 16 * 1024 * 1024

def is_tif(name):
    return str(name).lower().endswith(('.tif', '.tiff'))

def is_compressed_tif(name):
    return str(name).lower().endswith(('.tif' + SUFFIX, '.tiff' + SUFFIX))

def session_dirs(session_dir):
    """The session's raw TIFF folder and its subfolders (filter*, ref, ...)."""
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        return []
    return [session_dir] + sorted(d for d in session_dir.iterdir() if d.is_dir() and not d.name.startswith('.'))

def compress_file(tif, level=LEVEL, threads=8):
    """Compress one tif to tif.zst, verify the round trip, then delete the tif. Returns (original bytes, compressed bytes)."""
    import zstandard as zstd
    tif = Path(tif)
    out, tmp = tif.with_name(tif.name + SUFFIX), tif.with_name(tif.name + SUFFIX + '.tmp')
    st = tif.stat()
    h_in = hashlib.sha256()
    with open(tif, 'rb') as fin, open(tmp, 'wb') as fout:
        with zstd.ZstdCompressor(level=level, threads=threads, write_checksum=True).stream_writer(fout, size=st.st_size, closefd=False) as writer:
            while chunk := fin.read(CHUNK):
                h_in.update(chunk)
                writer.write(chunk)
    h_out, n = hashlib.sha256(), 0
    with open(tmp, 'rb') as fin, zstd.ZstdDecompressor().stream_reader(fin) as reader:
        while chunk := reader.read(CHUNK):
            h_out.update(chunk)
            n += len(chunk)
    if n != st.st_size or h_out.digest() != h_in.digest():
        tmp.unlink()
        raise IOError(f'Compressed copy of {tif} does not match the original; kept the original.')
    os.utime(tmp, (st.st_atime, st.st_mtime))  # keep the acquisition time
    tmp.replace(out)
    tif.unlink()
    return st.st_size, out.stat().st_size

def decompress_file(zst, out=None):
    """Decompress tif.zst back to tif (or to out), then delete the .zst if decompressing in place."""
    import zstandard as zstd
    zst = Path(zst)
    target = Path(out) if out else zst.with_name(zst.name[:-len(SUFFIX)])
    tmp = target.with_name(target.name + '.tmp')
    with open(zst, 'rb') as fin, open(tmp, 'wb') as fout:
        zstd.ZstdDecompressor().copy_stream(fin, fout, read_size=CHUNK, write_size=CHUNK)
    st = zst.stat()
    os.utime(tmp, (st.st_atime, st.st_mtime))
    tmp.replace(target)
    if out is None:
        zst.unlink()
    return target

def compress_dir(session_dir, workers=2, threads=8):
    """Compress every tif in the session folder and its subfolders. Returns a summary dict."""
    files = [f for d in session_dirs(session_dir) for f in sorted(d.iterdir()) if f.is_file() and is_tif(f.name)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        sizes = list(pool.map(lambda f: compress_file(f, threads=threads), files))
    before, after = sum(s[0] for s in sizes), sum(s[1] for s in sizes)
    summary = {'files': len(files), 'before_gb': before / 1e9, 'after_gb': after / 1e9, 'seconds': time.time() - t0}
    if files:
        print(f"Compressed {len(files)} TIFFs in {session_dir}: {summary['before_gb']:.1f} GB -> {summary['after_gb']:.1f} GB "
              f"({after / before:.0%}) in {summary['seconds'] / 60:.1f} min")
    else:
        print(f'No uncompressed TIFFs in {session_dir}.')
    return summary

def decompress_dir(directory, workers=4):
    """Decompress every tif.zst directly in directory (not subfolders) so tools that read TIFFs can use them."""
    directory = Path(directory)
    files = sorted(f for f in directory.iterdir() if f.is_file() and is_compressed_tif(f.name)) if directory.is_dir() else []
    if files:
        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(decompress_file, files))
        print(f'Decompressed {len(files)} TIFFs in {directory} ({(time.time() - t0) / 60:.1f} min)')
    return len(files)

@contextlib.contextmanager
def readable_tif(path):
    """A path ScanImage readers can open: path itself, or a temporary decompressed copy of a .tif.zst."""
    path = Path(path)
    if not is_compressed_tif(path.name):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix='mouse_imaging_') as tmpdir:
        yield decompress_file(path, out=Path(tmpdir) / path.name[:-len(SUFFIX)])
