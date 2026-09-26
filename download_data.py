#!/usr/bin/env python
"""Download the CT2Yarn volumes.

Two sources, pick whichever suits you:

  raw        18 NRRD, 27.35 GB, from Zenodo into data/raw/. The full scans, and
             what you need to reproduce the pipeline from step v0.
             https://doi.org/10.5281/zenodo.22822228
  processed  18 NRRD, 2.27 GB, from the GitHub release into data/processed/.
             Already Otsu-thresholded, so you can skip v0 and start at v2. Much
             the faster way in if you only want to run the reconstruction.

Only the standard library is used, so this runs before you install anything
else. Downloads resume: a file already at its published size is skipped, and a
partial file is continued with an HTTP range request.

    python download_data.py                    # raw, into data/raw
    python download_data.py --processed        # processed, into data/processed
    python download_data.py --sample bar G1    # just these
    python download_data.py --list             # show what is on offer
    python download_data.py --check            # see what is already on disk
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

RECORD = '22822228'
API = f'https://zenodo.org/api/records/{RECORD}'
DOI = 'https://doi.org/10.5281/zenodo.22822228'
REPO = 'chang9luo/ct2yarn'
TAG = 'data-processed-v1'
REL_API = f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}'
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch_record():
    with urllib.request.urlopen(API, timeout=60) as r:
        d = json.load(r)
    files = [{'name': f['key'],
              'size': int(f.get('size', 0)),
              'url': f['links']['self']}
             for f in d.get('files', [])]
    return d.get('metadata', {}), sorted(files, key=lambda x: x['name'])


def fetch_release():
    """The processed volumes live as GitHub release assets."""
    req = urllib.request.Request(
        REL_API, headers={'Accept': 'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            sys.exit(f'No published release {TAG} on {REPO}.\n'
                     f'If you are the maintainer it may still be a draft. '
                     f'Otherwise use the raw volumes from Zenodo:\n'
                     f'  python download_data.py')
        raise

    files = [{'name': a['name'],
              'size': int(a.get('size', 0)),
              'url': a['browser_download_url']}
             for a in d.get('assets', []) if a['name'].endswith('.nrrd')]
    meta = {'title': d.get('name') or TAG, 'license': 'CC-BY-4.0'}
    return meta, sorted(files, key=lambda x: x['name'])


def human(n):
    return f'{n / 1e9:.2f} GB' if n >= 1e9 else f'{n / 1e6:.1f} MB'


def download(f, dest, retries=4):
    """Fetch one file, resuming a partial download when possible."""
    tmp = dest + '.part'
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have > f['size']:
            os.remove(tmp)
            have = 0
        if have == f['size']:
            break
        req = urllib.request.Request(f['url'])
        if have:
            req.add_header('Range', f'bytes={have}-')
        try:
            with urllib.request.urlopen(req, timeout=120) as r, \
                 open(tmp, 'ab' if have else 'wb') as out:
                # a server that ignores Range restarts the file
                if have and r.status != 206:
                    out.close()
                    out = open(tmp, 'wb')
                    have = 0
                t0, last = time.time(), 0.0
                while True:
                    blk = r.read(8 << 20)
                    if not blk:
                        break
                    out.write(blk)
                    have += len(blk)
                    now = time.time()
                    if now - last > 0.5:
                        last = now
                        pct = 100.0 * have / f['size'] if f['size'] else 100.0
                        spd = have / max(now - t0, 1e-6) / 1e6
                        sys.stdout.write(f'\r    {pct:5.1f}%  {human(have)}'
                                         f' / {human(f["size"])}  {spd:6.1f} MB/s   ')
                        sys.stdout.flush()
            sys.stdout.write('\r' + ' ' * 70 + '\r')
            break
        except Exception as ex:
            sys.stdout.write('\n')
            print(f'    attempt {attempt}/{retries} failed: {str(ex)[:120]}')
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    got = os.path.getsize(tmp)
    if f['size'] and got != f['size']:
        raise RuntimeError(f'{f["name"]}: got {got} bytes, expected {f["size"]}')
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', action='store_true',
                    help='fetch the Otsu-thresholded volumes from the GitHub '
                         'release instead of the raw scans from Zenodo')
    ap.add_argument('--out-dir', default=None,
                    help='where the volumes go (default: data/raw, or '
                         'data/processed with --processed)')
    ap.add_argument('--sample', nargs='+', metavar='NAME',
                    help='only these samples, matched on the file stem')
    ap.add_argument('--list', action='store_true',
                    help='print what is on offer and exit')
    ap.add_argument('--check', action='store_true',
                    help='report which files are already complete, download nothing')
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(
            HERE, 'data', 'processed' if args.processed else 'raw')

    if args.processed:
        print(f'Source: github.com/{REPO} release {TAG}')
        try:
            meta, files = fetch_release()
        except SystemExit:
            raise
        except Exception as ex:
            sys.exit(f'Could not reach GitHub: {ex}')
    else:
        print(f'Source: {DOI}')
        try:
            meta, files = fetch_record()
        except Exception as ex:
            sys.exit(f'Could not reach Zenodo: {ex}')
    lic = meta.get('license')
    print(f'  {meta.get("title", "")}')
    print(f'  {len(files)} files, {human(sum(f["size"] for f in files))}, '
          f'{lic.get("id") if isinstance(lic, dict) else lic}')

    if args.sample:
        want = set(args.sample)
        files = [f for f in files if f['name'].rsplit('.', 1)[0] in want]
        found = {f['name'].rsplit('.', 1)[0] for f in files}
        for miss in sorted(want - found):
            print(f'  WARNING: no such sample: {miss}')
        if not files:
            sys.exit('Nothing matched.')

    if args.list:
        print()
        for f in files:
            print(f'  {f["name"]:32s} {human(f["size"]):>10s}')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Target: {args.out_dir}\n')

    ok = skipped = 0
    for i, f in enumerate(files, 1):
        dest = os.path.join(args.out_dir, f['name'])
        head = f'[{i:2d}/{len(files)}] {f["name"]:32s} {human(f["size"]):>10s}'
        if os.path.exists(dest) and os.path.getsize(dest) == f['size']:
            print(f'{head}  already present')
            skipped += 1
            continue
        if args.check:
            state = 'INCOMPLETE' if os.path.exists(dest) else 'MISSING'
            print(f'{head}  {state}')
            continue
        print(head)
        download(f, dest)
        print(f'{" " * 11}done')
        ok += 1

    if args.check:
        print('\nCheck complete.')
    else:
        print(f'\n{ok} downloaded, {skipped} already there. '
              f'Next: bash run_preprocess.sh')


if __name__ == '__main__':
    main()
