"""
Prints the decompiled C of functions of game.exe with a line of the registry above each one: the status,
the domain, the callers which are implemented and the Python which implements it.

    PYTHONPATH=src venv/bin/python -m tools.func_dump 5aea80 5d53b0      functions by their addresses
    PYTHONPATH=src venv/bin/python -m tools.func_dump 5616c0-561b00     every function of a block
    PYTHONPATH=src venv/bin/python -m tools.func_dump --brief 4e1770-4e59f0   only the registry lines
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import re
import sys
import json
import argparse

#------------------------------------------------------------------------------

REMAKE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

HEADER_RE = re.compile(r'^//==== (\S+) @ ([0-9a-f]{8})$', re.M)

#------------------------------------------------------------------------------


def load_registry(json_file_path):
    with open(json_file_path, encoding='utf-8') as f:
        registry = json.load(f)
    return {int(e['address'], 16): e for e in registry['functions']}


def read_bodies(c_file_path, wanted):
    with open(c_file_path, encoding='latin-1') as f:
        text = f.read()
    headers = list(HEADER_RE.finditer(text))
    bodies = {}
    for i, m in enumerate(headers):
        address = int(m.group(2), 16)
        if address in wanted:
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            bodies[address] = text[m.start():end].rstrip() + '\n'
    return bodies


def registry_line(entry, functions):
    parts = ['%s %s' % (entry['address'], entry['name'])]
    if entry.get('class'):
        parts.append('%s::%s' % (entry['class'], entry['method']))
    elif entry.get('guess_class'):
        parts.append('class? %s' % entry['guess_class'])
    parts.append(entry['status'])
    if entry.get('domain'):
        parts.append('%s/%s' % (entry['domain'], entry['tier']))
    if 'lines' in entry:
        parts.append('%d lines' % entry['lines'])
    if 'callers' in entry:
        parts.append('callers %d calls %d' % (entry['callers'], entry['calls']))
    if entry.get('covered_by'):
        parts.append('covered by ' + ' '.join(entry['covered_by'][:6]))
    for p in entry.get('python', [])[:4]:
        parts.append('py %s:%d %s' % (p['file'][len('src/'):], p['line'], p['symbol']))
    if entry.get('strings'):
        parts.append('strings %s' % ' '.join(entry['strings'][:4]))
    return '// ' + ' | '.join(parts)


def shorten(body, max_lines):
    if not max_lines:
        return body
    lines = body.split('\n')
    # the declarations of the locals of Ghidra say nothing: "  int iVar1;"
    kept = [l for l in lines if l.strip() not in ('', '{', '}') and
            (l.strip().startswith('return') or not re.match(r'^  [\w *]+ [*]*\w+(\[\w+\])?;$', l))]
    if len(kept) > max_lines:
        kept = kept[:max_lines] + ['  ... %d more lines' % (len(kept) - max_lines)]
    return '\n'.join(kept) + '\n'


def select(functions, specs):
    addresses = []
    for spec in specs:
        if '-' in spec:
            lo, hi = [int(x, 16) for x in spec.split('-', 1)]
            addresses.extend(a for a in sorted(functions) if lo <= a <= hi)
        else:
            addresses.append(int(spec, 16))
    return addresses


def main():
    parser = argparse.ArgumentParser(description='decompiled C of game.exe functions with their registry lines')
    parser.add_argument('specs', nargs='+', help='addresses (5aea80) or blocks (5616c0-561b00)')
    parser.add_argument('--brief', action='store_true', help='only the registry lines')
    parser.add_argument('--max-lines', dest='max_lines', type=int, default=0,
                        help='cut every body after this many lines (the declarations of locals are left out)')
    parser.add_argument('--registry', dest='json_file_path', metavar='JSON_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 'func_registry.json'))
    parser.add_argument('--c', dest='c_file_path', metavar='C_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 'build', 'ghidra', 'game_exe.c'))
    args = parser.parse_args()
    functions = load_registry(args.json_file_path)
    addresses = select(functions, args.specs)
    bodies = {} if args.brief else read_bodies(args.c_file_path, set(addresses))
    out = []
    for address in addresses:
        entry = functions.get(address)
        if entry is None:
            out.append('// %06x: no such function\n' % address)
            continue
        out.append(registry_line(entry, functions) + '\n')
        if not args.brief:
            out.append(shorten(bodies.get(address, ''), args.max_lines) + '\n')
    sys.stdout.write(''.join(out))


if __name__ == '__main__':
    main()
