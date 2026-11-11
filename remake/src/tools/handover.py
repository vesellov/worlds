"""
Sharing the remake with a second developer without git (remake/ is not in the repository), see PLAN_DEV2.md.

    PYTHONPATH=src venv/bin/python -m tools.handover snapshot handover/base.tar.gz
    PYTHONPATH=src venv/bin/python -m tools.handover pack handover/base.tar.gz handover/dev2.tar.gz
    PYTHONPATH=src venv/bin/python -m tools.handover merge handover/base.tar.gz handover/dev2.tar.gz [--apply]

snapshot: the base version both developers start from, the project files with MANIFEST.sha256.
pack (second developer): the files added, changed or deleted against the base, with CHANGES.txt.
merge: every file of the pack against the base and the current tree; a file changed on one side only is taken
from that side, a file changed on both sides goes through "git merge-file" (conflicts are marked in the file).
Without --apply it only prints what it would do; with --apply the touched files are saved first
to handover/backup-<time>.tar.gz.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

#------------------------------------------------------------------------------

ROOT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# what belongs to the project: folders and single files relative to remake/
PROJECT_DIR_NAMES = ('src', 'tests', 're')
PROJECT_FILE_NAMES = ('Makefile', 'requirements.txt', 'README.md', 'first_scan.md', 'PLAN_DEV2.md',
                      'NOTES_DEV2.md', 'first_scan_dev2.md')
SKIPPED_DIR_NAMES = ('__pycache__', 'venv', 'build', 'handover')
SKIPPED_SUFFIXES = ('.pyc', '.pyo')

MANIFEST_NAME = 'MANIFEST.sha256'
CHANGES_NAME = 'CHANGES.txt'
HANDOVER_DIR_NAME = 'handover'

#------------------------------------------------------------------------------


def project_files(root_dir_path):
    """
    The paths of the project files relative to root_dir_path, with "/" between the folders.
    """
    result = []
    for dir_name in PROJECT_DIR_NAMES:
        for dir_path, sub_dir_names, file_names in os.walk(os.path.join(root_dir_path, dir_name)):
            sub_dir_names[:] = sorted(d for d in sub_dir_names if d not in SKIPPED_DIR_NAMES)
            for file_name in sorted(file_names):
                if file_name.endswith(SKIPPED_SUFFIXES):
                    continue
                file_path = os.path.join(dir_path, file_name)
                result.append(os.path.relpath(file_path, root_dir_path).replace(os.sep, '/'))
    for file_name in PROJECT_FILE_NAMES:
        if os.path.isfile(os.path.join(root_dir_path, file_name)):
            result.append(file_name)
    return result


def file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def tree_hashes(root_dir_path):
    return {name: file_hash(os.path.join(root_dir_path, name)) for name in project_files(root_dir_path)}


def manifest_text(hashes):
    return ''.join('%s  %s\n' % (hashes[name], name) for name in sorted(hashes))


def parse_manifest(text):
    hashes = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split('  ', 1)
            hashes[name] = digest
    return hashes


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(data))


def read_member(archive, name):
    try:
        member = archive.getmember(name)
    except KeyError:
        return None
    return archive.extractfile(member).read()


def read_file(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'rb') as f:
        return f.read()


def snapshot(archive_path):
    hashes = tree_hashes(ROOT_DIR_PATH)
    with tarfile.open(archive_path, 'w:gz') as archive:
        add_bytes(archive, MANIFEST_NAME, manifest_text(hashes).encode())
        for name in sorted(hashes):
            archive.add(os.path.join(ROOT_DIR_PATH, name), arcname='files/' + name)
    print('base: %d files -> %s' % (len(hashes), archive_path))


def pack(base_archive_path, pack_archive_path):
    with tarfile.open(base_archive_path, 'r:gz') as base:
        base_hashes = parse_manifest(read_member(base, MANIFEST_NAME).decode())
    hashes = tree_hashes(ROOT_DIR_PATH)
    added = sorted(set(hashes) - set(base_hashes))
    deleted = sorted(set(base_hashes) - set(hashes))
    changed = sorted(n for n in set(hashes) & set(base_hashes) if hashes[n] != base_hashes[n])
    lines = ['A %s' % n for n in added] + ['M %s' % n for n in changed] + ['D %s' % n for n in deleted]
    with tarfile.open(pack_archive_path, 'w:gz') as archive:
        add_bytes(archive, MANIFEST_NAME, manifest_text(base_hashes).encode())
        add_bytes(archive, CHANGES_NAME, ''.join(line + '\n' for line in lines).encode())
        for name in added + changed:
            archive.add(os.path.join(ROOT_DIR_PATH, name), arcname='files/' + name)
    print('\n'.join(lines))
    print('pack: %d added, %d changed, %d deleted -> %s' % (len(added), len(changed), len(deleted),
                                                           pack_archive_path))


def merge_text(mine, base, theirs, temp_dir_path):
    """
    The result of "git merge-file" and the number of conflicts in it.
    """
    paths = []
    for name, data in (('mine', mine), ('base', base), ('theirs', theirs)):
        file_path = os.path.join(temp_dir_path, name)
        with open(file_path, 'wb') as f:
            f.write(data or b'')
        paths.append(file_path)
    done = subprocess.run(['git', 'merge-file', '-p', '-L', 'mine', '-L', 'base', '-L', 'dev2'] + paths,
                          capture_output=True)
    if done.returncode < 0 or done.returncode > 127:
        sys.exit('git merge-file failed: %s' % done.stderr.decode(errors='replace'))
    return done.stdout, done.returncode


def plan_merge(base_archive_path, pack_archive_path, temp_dir_path):
    """
    The list of (action, name, data) for the files of the pack: "take", "delete", "merged", "conflict",
    "keep" (nothing to do).
    """
    with tarfile.open(base_archive_path, 'r:gz') as base, tarfile.open(pack_archive_path, 'r:gz') as pack_archive:
        base_hashes = parse_manifest(read_member(base, MANIFEST_NAME).decode())
        pack_hashes = parse_manifest(read_member(pack_archive, MANIFEST_NAME).decode())
        if base_hashes != pack_hashes:
            sys.exit('the pack was made against another base')
        changes = read_member(pack_archive, CHANGES_NAME).decode().splitlines()
        actions = []
        for line in changes:
            kind, name = line.split(' ', 1)
            base_data = read_member(base, 'files/' + name)
            mine = read_file(os.path.join(ROOT_DIR_PATH, name))
            theirs = None if kind == 'D' else read_member(pack_archive, 'files/' + name)
            if mine == theirs:
                actions.append(('keep', name, None))
            elif mine == base_data:
                actions.append(('delete' if theirs is None else 'take', name, theirs))
            elif theirs is None:
                actions.append(('conflict', name, mine))
            else:
                data, conflicts = merge_text(mine, base_data, theirs, temp_dir_path)
                actions.append(('conflict' if conflicts else 'merged', name, data))
    return actions


def backup(names):
    dir_path = os.path.join(ROOT_DIR_PATH, HANDOVER_DIR_NAME)
    os.makedirs(dir_path, exist_ok=True)
    backup_file_path = os.path.join(dir_path, 'backup-%s.tar.gz' % time.strftime('%Y%m%d-%H%M%S'))
    with tarfile.open(backup_file_path, 'w:gz') as archive:
        for name in names:
            file_path = os.path.join(ROOT_DIR_PATH, name)
            if os.path.isfile(file_path):
                archive.add(file_path, arcname=name)
    return backup_file_path


def merge(base_archive_path, pack_archive_path, apply):
    temp_dir_path = tempfile.mkdtemp(prefix='handover-')
    try:
        actions = plan_merge(base_archive_path, pack_archive_path, temp_dir_path)
    finally:
        shutil.rmtree(temp_dir_path)
    for action, name, data in actions:
        print('%-8s %s' % (action, name))
    if not apply:
        print('dry run, --apply writes the files')
        return
    touched = [name for action, name, data in actions if action != 'keep']
    print('backup: %s' % backup(touched))
    for action, name, data in actions:
        file_path = os.path.join(ROOT_DIR_PATH, name)
        if action == 'delete':
            os.remove(file_path)
        elif action in ('take', 'merged') or (action == 'conflict' and data is not None):
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(data)
        if _Debug:
            print('wrote %s %s' % (action, name))
    conflicts = [name for action, name, data in actions if action == 'conflict']
    if conflicts:
        print('conflicts to resolve by hand (<<<<<<< mine / >>>>>>> dev2, or deleted by dev2 and changed here):')
        for name in conflicts:
            print('  ' + name)


def main():
    parser = argparse.ArgumentParser(description='sharing the remake with a second developer')
    commands = parser.add_subparsers(dest='command', required=True)
    snapshot_parser = commands.add_parser('snapshot')
    snapshot_parser.add_argument('archive_path', metavar='BASE_ARCHIVE')
    pack_parser = commands.add_parser('pack')
    pack_parser.add_argument('base_archive_path', metavar='BASE_ARCHIVE')
    pack_parser.add_argument('pack_archive_path', metavar='PACK_ARCHIVE')
    merge_parser = commands.add_parser('merge')
    merge_parser.add_argument('base_archive_path', metavar='BASE_ARCHIVE')
    merge_parser.add_argument('pack_archive_path', metavar='PACK_ARCHIVE')
    merge_parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.command == 'snapshot':
        snapshot(args.archive_path)
    elif args.command == 'pack':
        pack(args.base_archive_path, args.pack_archive_path)
    else:
        merge(args.base_archive_path, args.pack_archive_path, args.apply)


if __name__ == '__main__':
    main()
