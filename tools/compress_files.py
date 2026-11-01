"""
Compresses game data files with gzip and xz before they are published on GitHub, the client unpacks them in src/client/loader.py.

Both formats are stored next to each other. xz is smaller, but the lzma module is optional in some Python builds
(python-for-android, kivy-ios, Pyodide), while gzip only needs zlib which every Python build has.
The client downloads the format set in its configuration and falls back to gzip when lzma is not available.

Every source file becomes <name>.gz and <name>.xz, the files are never cut into parts.
Folders are compressed file by file, their sub folders are kept.

The output does not change when the same files are compressed again (no timestamp is stored),
so re-running the tool does not produce new git changes.

    python3 tools/compress_files.py ~/.worlds/data/figures.res ~/.worlds/data/textures.res --out ./compressed/
    python3 tools/compress_files.py catalog assets --out ./compressed/
    python3 tools/compress_files.py catalog --out ./compressed/ --formats gz
"""

import os
import sys
import gzip
import lzma
import shutil
import hashlib
import argparse

CHUNK_SIZE = 1024 * 1024

FORMATS = ('gz', 'xz')

# GitHub refuses files over 100 MB, the tool only warns, the file is still written
GITHUB_MAX_SIZE = 100 * 1024 * 1024


def open_compressed(fmt, fileobj, mode):
    if fmt == 'gz':
        # empty file name and zero mtime keep the output identical between runs
        return gzip.GzipFile(filename='', mode=mode, compresslevel=9, fileobj=fileobj, mtime=0)
    if mode == 'rb':
        return lzma.LZMAFile(fileobj, mode='rb')
    # xz stores no time stamp, CRC64 is checked by the client on decompression
    return lzma.LZMAFile(fileobj, mode=mode, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9 | lzma.PRESET_EXTREME)


def compress_file(source_path, dest_path, fmt):
    """
    Writes dest_path + '.<fmt>' and returns its path.
    """
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    out_path = f'{dest_path}.{fmt}'
    for old in old_parts(dest_path, fmt):
        os.remove(old)
    with open(source_path, 'rb') as src, open(out_path, 'wb') as raw:
        with open_compressed(fmt, raw, 'wb') as dst:
            shutil.copyfileobj(src, dst, CHUNK_SIZE)
    return out_path


def old_parts(dest_path, fmt):
    # <name>.<fmt>.0, <name>.<fmt>.1 ... left from older versions of the tool which cut big files into parts
    folder = os.path.dirname(dest_path) or '.'
    prefix = f'{os.path.basename(dest_path)}.{fmt}.'
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, n) for n in os.listdir(folder) if n.startswith(prefix) and n[len(prefix):].isdigit()]


def file_hash(f):
    h = hashlib.sha256()
    while True:
        data = f.read(CHUNK_SIZE)
        if not data:
            return h.hexdigest()
        h.update(data)


def verify(expected_hash, out_path, fmt):
    # decompress the written file again and compare with the source
    with open(out_path, 'rb') as raw, open_compressed(fmt, raw, 'rb') as f:
        return file_hash(f) == expected_hash


def is_compressed_output(name):
    # <name>.gz and <name>.xz (and old parts <name>.gz.N) are skipped when a folder is compressed again
    pieces = name.split('.')
    if pieces[-1].isdigit() and len(pieces) > 2:
        pieces = pieces[:-1]
    return len(pieces) > 1 and pieces[-1] in FORMATS


def collect(sources):
    # (source file, output name relative to --out)
    result = []
    for source in sources:
        source = os.path.abspath(os.path.expanduser(source))
        if os.path.isfile(source):
            result.append((source, os.path.basename(source)))
        elif os.path.isdir(source):
            base = os.path.dirname(source)
            for root, _, files in os.walk(source):
                for name in sorted(files):
                    if is_compressed_output(name):
                        continue
                    path = os.path.join(root, name)
                    result.append((path, os.path.relpath(path, base)))
        else:
            print(f'not found: {source}')
            sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description='compress game data files with gzip and xz for downloading by the client')
    parser.add_argument('sources', nargs='+', help='files or folders to compress')
    parser.add_argument('--out', required=True, help='output folder')
    parser.add_argument('--formats', default=','.join(FORMATS), help=f'comma separated list of formats to write (default {",".join(FORMATS)})')
    args = parser.parse_args()
    formats = [f.strip() for f in args.formats.split(',') if f.strip()]
    for fmt in formats:
        if fmt not in FORMATS:
            parser.error(f'unknown format: {fmt}, known formats: {", ".join(FORMATS)}')
    total_raw = 0
    total_compressed = {fmt: 0 for fmt in formats}
    errors = 0
    for source_path, rel_name in collect(args.sources):
        raw = os.path.getsize(source_path)
        total_raw += raw
        with open(source_path, 'rb') as f:
            expected_hash = file_hash(f)
        line = []
        for fmt in formats:
            out_path = compress_file(source_path, os.path.join(args.out, rel_name), fmt)
            compressed = os.path.getsize(out_path)
            total_compressed[fmt] += compressed
            ok = verify(expected_hash, out_path, fmt)
            if not ok:
                errors += 1
            warning = ', WARNING: over 100 MB, GitHub will refuse it' if compressed > GITHUB_MAX_SIZE else ''
            line.append(f'{"" if ok else "ERROR, content differs: "}{fmt} {compressed / 1e6:.2f} MB ({compressed / max(raw, 1):.0%}){warning}')
        print(f'{rel_name}: {raw / 1e6:.2f} MB -> ' + ', '.join(line))
    for fmt in formats:
        print(f'total {fmt}: {total_raw / 1e6:.1f} MB -> {total_compressed[fmt] / 1e6:.1f} MB ({total_compressed[fmt] / max(total_raw, 1):.0%}) in {args.out}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
