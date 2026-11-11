"""
Builds the registry of the functions of game.exe and of their implementations in the remake.

The functions are taken from the decompiled C of Ghidra (build/ghidra/game_exe.c, "//==== name @ address" headers).
The remake points to the original code in its comments and docstrings: an address of game.exe ("45ea20",
"0x50aaf0", "FUN_00461cb0") or a name of re/names.json ("CUnitPrototype::Read"). Every such reference is given
to the def or class it stands in (or the def right under a comment block), so every C function gets the list
of the Python symbols which implement it.

Statuses:
    implemented  referenced from a function or a class of the remake
    partial      referenced only as "<address> (partly)": the remake does a part of the function
    module       referenced only by the docstring of a module or by a module constant (a table, a number)
    indirect     not referenced, but called directly by an implemented function
    missing      game code without any reference
    library      CRT, MFC and the imports of DLLs linked into game.exe
    compiler     the exception handlers made by the compiler (Unwind@, Catch@)

    PYTHONPATH=src venv/bin/python -m tools.func_registry     (writes func_registry.json and FUNC_REGISTRY.md)
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import re
import ast
import json
import argparse
import tokenize
import collections

#------------------------------------------------------------------------------

from tools import func_domains
from tools import func_coverage

#------------------------------------------------------------------------------

REMAKE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# the static libraries (CRT, MFC) and the imports of DLLs start here, all game code is below
LIBRARY_START = 0x6e60a2

HEADER_RE = re.compile(r'^//==== (\S+) @ ([0-9a-f]{8})$', re.M)
CALL_RE = re.compile(r'\b([A-Za-z_~][\w@]*)\s*\(')
STRING_RE = re.compile(r'"((?:[^"\\\n]|\\.){3,60})"')
LABEL_RE = re.compile(r'\bs_([A-Za-z0-9_]+?)_00[0-9a-f]{6}\b')
PY_ADDR_RE = re.compile(r'(?<![0-9A-Za-z_#])(FUN_|0x)?(?:00)?([4-6][0-9a-fA-F]{5})(?![0-9A-Za-z_])')
PY_NAME_RE = re.compile(r'\b([A-Za-z_]\w*::~?\w+)')
PARTLY_RE = re.compile(r'^\)?[\s,]*\(?part(ly|ial)\b', re.I)
DOC_ADDR_RE = re.compile(r'(?<![0-9A-Za-z_#])(?:FUN_|0x)?(?:00)?([4-6][0-9a-f]{5})(?![0-9A-Za-z_])')

LIBRARY_NAMES = {
    'entry', 'RtlUnwind', 'Find', 'Close', 'GetOwner', 'InitString', 'FreeExtra', 'RemoveAll', 'AddHead', 'AddTail',
    'CCmdUI', 'Detach', 'ActivateTopParent', 'OnDevModeChange', 'OnDisplayChange', 'CRect', 'CMap<>',
}

# the shapes of the code which need no port: (name, status, note)
LOCAL_RE = re.compile(r'^[\w *]+ [*]*\w+(\[\w+\])?;$')
GETTER_RE = re.compile(r'^return (\*\()?\(?[\w ]*\*?\)?\(?(param_1|\*param_1)( \+ (0x)?[0-9a-f]+)?\)?;$')
CONST_RE = re.compile(r'^return (0x[0-9a-f]+|\d+|true|false|param_1|&DAT_\w+|DAT_\w+|s_\w+);$')
SETTER_RE = re.compile(r'^\*\(.*\)\(param_1 \+ 0x[0-9a-f]+\) = param_2;$')
FIELDS_SET_RE = re.compile(r'^\*\([\w *]+\)\(param_1 \+ (0x[0-9a-f]+|\d+)\) = '
                           r'(param_\d|\*param_\d|param_\d\[\d\]|0x[0-9a-f]+|-?\d+|\([\w *]+\)param_\d);$|'
                           r'^param_1\[\d+\] = (param_\d|\*param_\d|param_\d\[\d\]|0x[0-9a-f]+|-?\d+);$|'
                           r'^\*param_1 = (param_\d|\*param_\d|param_\d\[\d\]|0x[0-9a-f]+|-?\d+);$')
BITS_RE = re.compile(r'^\*\((uint|int) \*\)\(param_1 \+ (0x[0-9a-f]+|\d+)\) = '
                     r'\*\((uint|int) \*\)\(param_1 \+ \2\) (\| param_2|& ~param_2|& 0x[0-9a-f]+|\| 0x[0-9a-f]+);$')
LIST_NEXT_LINES = ['piVar1 = (int *)*param_1;', '*param_1 = *piVar1;', 'return (int)(piVar1 + 2);']
ITER_COPY_RE = re.compile(r'^\*param_2 = (\*param_1|\*\(undefined4 \*\)\*param_1);$')
LIST_INSERT_RE = re.compile(r'^piVar1 = \(int \*\)FUN_[0-9a-f]{8}\(param_3\);$')
LIST_INSERT_LINES = ['*piVar1 = param_2;', 'piVar1[1] = *(int *)(param_2 + 4);',
                     '**(int **)(param_2 + 4) = (int)piVar1;', '*(int **)(param_2 + 4) = piVar1;',
                     '*param_1 = (int)piVar1;', 'return param_1;']
ALIGNED_NEW_RE = re.compile(r'^iVar1 = operator_new\(param_1 \* \d+ \+ 0x24\);$')
DELETING_DTOR_LINES = ['if ((param_2 & 1) != 0) {', 'operator_delete(param_1);', 'return param_1;']
ITER_NEXT_LINES = ['puVar1 = *(undefined4 **)(param_1 + 4);', 'if (puVar1 == (undefined4 *)0x0) {', 'return 0;',
                   '*(undefined4 *)(param_1 + 4) = *puVar1;', 'return puVar1[2];']
VECTOR_AT_RE = re.compile(r'^if \(param_2 < \(uint\)\(\(param_1\[1\] - \*param_1\) / 0x[0-9a-f]+\)\) \{$')

# a phase of the trace shorter than this (in windows of 2 s) is a load or a click, not a run of ticks
PERIODIC_MIN_WINDOWS = 5

STATUS_ORDER = ('implemented', 'partial', 'module', 'builtin', 'indirect', 'missing', 'later', 'skip', 'library', 'compiler')

# the decisions of re/func_decisions.json: a reference from Python wins over them
DECISIONS = ('builtin', 'later', 'skip')

#------------------------------------------------------------------------------


class Function(object):

    def __init__(self, name, address, line):
        self.name = name
        self.address = address
        self.line = line
        self.lines = 0
        self.size = 0
        self.kind = 'game'
        self.class_name = None
        self.method_name = None
        self.calls = set()
        self.callers = set()
        self.strings = []
        self.python = []
        self.docs = []
        self.covered_by = []
        self.status = None
        self.guess_class = None
        self.all_strings = set()
        self.apis = set()
        self.domain = None
        self.domain_source = None
        self.decision_note = None
        self.pattern = None
        # {phase: windows} of the traces of re/coverage, and the phases where the function runs all the time
        self.coverage = {}
        self.periodic = []

    def title(self):
        if self.class_name:
            return '%s::%s' % (self.class_name, self.method_name)
        return self.name

    def to_json(self):
        entry = {
            'address': '%06x' % self.address,
            'name': self.name,
            'status': self.status,
        }
        if self.kind == 'compiler':
            return entry
        entry['lines'] = self.lines
        entry['size'] = self.size
        if self.kind == 'library':
            return entry
        if self.class_name:
            entry['class'] = self.class_name
            entry['method'] = self.method_name
        elif self.guess_class:
            entry['guess_class'] = self.guess_class
        entry['callers'] = len(self.callers)
        entry['calls'] = len(self.calls)
        if self.python:
            entry['python'] = self.python
        if self.domain:
            entry['domain'] = self.domain
            entry['domain_source'] = self.domain_source
            entry['tier'] = func_domains.DOMAINS[self.domain][0]
        if self.decision_note:
            entry['note'] = self.decision_note
        if self.covered_by:
            entry['covered_by'] = ['%06x' % a for a in self.covered_by]
        if self.coverage:
            entry['coverage'] = self.coverage
        if self.periodic:
            entry['periodic'] = self.periodic
        if self.docs:
            entry['docs'] = self.docs
        if self.strings:
            entry['strings'] = self.strings
        return entry

#------------------------------------------------------------------------------


def classify(name, address):
    if name.startswith(('Unwind@', 'Catch@')):
        return 'compiler'
    if address >= LIBRARY_START:
        return 'library'
    if name.startswith(('_', '~', 'Afx', 'Ordinal_', 'operator', '`', 'crt_')) or name in LIBRARY_NAMES:
        return 'library'
    return 'game'


def code_pattern(body):
    # body: the text after the header line of a function
    lines = [l.strip() for l in body.split('{', 1)[-1].split('\n')]
    # "return 0x52;" has the shape of a declaration too
    lines = [l for l in lines if l and l not in ('{', '}') and (l.startswith('return') or not LOCAL_RE.match(l))]
    if lines == ['return;']:
        return 'skip', 'auto: empty function'
    if len(lines) == 1 and GETTER_RE.match(lines[0]):
        return 'builtin', 'auto: getter of a field'
    if len(lines) == 1 and CONST_RE.match(lines[0]):
        return 'builtin', 'auto: returns a constant'
    if len(lines) == 2 and lines[1] == 'return;' and SETTER_RE.match(lines[0]):
        return 'builtin', 'auto: setter of a field'
    if len(lines) == 2 and lines[1] == 'return;' and BITS_RE.match(lines[0]):
        return 'builtin', 'auto: sets or clears bits of a field'
    if 2 <= len(lines) <= 9 and lines[-1] == 'return;' and all(FIELDS_SET_RE.match(l) for l in lines[:-1]):
        return 'builtin', 'auto: setter of fields'
    if lines == LIST_NEXT_LINES:
        return 'builtin', 'auto: the next node of a list'
    if 2 <= len(lines) <= 4 and VECTOR_AT_RE.match(lines[0]):
        return 'builtin', 'auto: vector::at, a list index'
    if len(lines) == 2 and lines[1] == 'return;' and ITER_COPY_RE.match(lines[0]):
        return 'builtin', 'auto: copy of an iterator'
    if len(lines) == 7 and LIST_INSERT_RE.match(lines[0]) and lines[1:] == LIST_INSERT_LINES:
        return 'builtin', 'auto: insert into a list'
    if len(lines) == 3 and ALIGNED_NEW_RE.match(lines[0]):
        return 'builtin', 'auto: aligned array allocation'
    if len(lines) == 4 and re.match(r'^FUN_[0-9a-f]{8}\(\);$', lines[0]) and lines[1:] == DELETING_DTOR_LINES:
        return 'builtin', 'auto: deleting destructor'
    if lines == ITER_NEXT_LINES:
        return 'builtin', 'auto: the next item of a list iterator'
    if 'FUN_004298d0()' in body and '% (uint)' in body:
        # 4298d0 is the hash of a string (h = h * 5 + c) of the hash_map of STLport
        return 'builtin', 'auto: hash_map of strings, a dict'
    return None


def read_c(c_file_path, names):
    with open(c_file_path, encoding='latin-1') as f:
        text = f.read()
    headers = list(HEADER_RE.finditer(text))
    functions = []
    for i, m in enumerate(headers):
        name, address = m.group(1), int(m.group(2), 16)
        func = Function(name, address, 0)
        func.kind = classify(name, address)
        full_name = names.get('%x' % address)
        if full_name and '::' in full_name:
            func.class_name, func.method_name = full_name.split('::', 1)
        functions.append((func, m.start(), headers[i + 1].start() if i + 1 < len(headers) else len(text)))
    # the line numbers, counted once over the whole text
    line = 1
    pos = 0
    for func, start, end in functions:
        line += text.count('\n', pos, start)
        pos = start
        func.line = line
        func.lines = text.count('\n', start, end)
    by_name = collections.defaultdict(list)
    for func, _, _ in functions:
        by_name[func.name].append(func)
    unique = {n: fs[0] for n, fs in by_name.items() if len(fs) == 1}
    for func, start, end in functions:
        if func.kind != 'game':
            continue
        body = text[start:end]
        body = body[body.find('\n'):]
        for call in CALL_RE.findall(body):
            callee = unique.get(call)
            if callee is None:
                if call not in by_name:
                    # an import of a DLL called by its name (timeGetTime, _AIL_start_sample_4)
                    func.apis.add(call)
            elif callee is not func:
                func.calls.add(callee.address)
                if callee.kind == 'library':
                    func.apis.add(callee.name)
        strings = []
        for s in STRING_RE.findall(body) + LABEL_RE.findall(body):
            if s not in strings:
                strings.append(s)
        func.all_strings = set(strings)
        func.pattern = code_pattern(body)
        func.strings = strings[:8]
    result = [f for f, _, _ in functions]
    result.sort(key=lambda f: f.address)
    for i, func in enumerate(result):
        if i + 1 < len(result):
            func.size = result[i + 1].address - func.address
    return result


def symbol_ranges(tree):
    ranges = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualname = prefix + child.name
                first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                ranges.append((first, child.end_lineno, qualname, 'code'))
                walk(child, qualname + '.')
            else:
                walk(child, prefix)

    walk(tree, '')
    # the constants of the module: a table or a number taken from game.exe
    for child in tree.body:
        targets = []
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        target_names = [t.id for t in targets if isinstance(t, ast.Name)]
        if target_names:
            ranges.append((child.lineno, child.end_lineno, target_names[0], 'constant'))
    return ranges


def find_symbol(ranges, lines, line_no):
    best = None
    for first, last, qualname, kind in ranges:
        if first <= line_no <= last:
            if best is None or first >= best[0]:
                best = (first, qualname, kind)
    if best is not None:
        return best[1], best[2]
    # a block of comments right above a def, a class or a constant belongs to it
    if not lines[line_no - 1].strip().startswith('#'):
        return None, None
    i = line_no
    while i < len(lines) and lines[i].strip().startswith('#') and not lines[i].strip().startswith('#---'):
        i += 1
    for first, last, qualname, kind in ranges:
        if first == i + 1:
            return qualname, kind
    return None, None


def read_python(py_file_path, rel_path, functions_by_address, starts, names_by_full):
    with open(py_file_path, encoding='utf-8') as f:
        source = f.read()
    lines = source.split('\n')
    tree = ast.parse(source)
    ranges = symbol_ranges(tree)
    refs = []
    unresolved = []
    with open(py_file_path, 'rb') as f:
        tokens = list(tokenize.tokenize(f.readline))
    for tok in tokens:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        for offset, text_line in enumerate(tok.string.split('\n')):
            line_no = tok.start[0] + offset
            for m in PY_ADDR_RE.finditer(text_line):
                prefix, digits = m.group(1), m.group(2).lower()
                address = int(digits, 16)
                is_hex = bool(prefix) or any(c in 'abcdef' for c in digits)
                func = functions_by_address.get(address)
                inside = False
                if func is None:
                    if not is_hex:
                        continue
                    func = containing(starts, functions_by_address, address)
                    inside = True
                    if func is None:
                        unresolved.append((rel_path, line_no, digits))
                        continue
                # "5641f0 (partly)": the Python does only a part of the function
                partly = bool(PARTLY_RE.match(text_line[m.end():]))
                refs.append((func, line_no, 'address', inside, partly))
            for m in PY_NAME_RE.finditer(text_line):
                address = names_by_full.get(m.group(1))
                if address is not None and address in functions_by_address:
                    refs.append((functions_by_address[address], line_no, 'name', False,
                                 bool(PARTLY_RE.match(text_line[m.end():]))))
    result = []
    for func, line_no, via, inside, partly in refs:
        symbol, kind = find_symbol(ranges, lines, line_no)
        item = {
            'file': rel_path,
            'symbol': symbol or '<module>',
            'line': line_no,
            'via': via,
        }
        if kind == 'constant':
            item['constant'] = True
        if inside:
            item['inside'] = True
        if partly:
            item['partly'] = True
        result.append((func, item))
    return result, unresolved


def containing(starts, functions_by_address, address):
    lo, hi = 0, len(starts) - 1
    if address < starts[0]:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= address:
            lo = mid
        else:
            hi = mid - 1
    func = functions_by_address[starts[lo]]
    if func.kind != 'game' or address >= func.address + max(func.size, 1):
        return None
    return func


def read_docs(doc_file_path, functions_by_address):
    rel_name = os.path.basename(doc_file_path)
    section = ''
    with open(doc_file_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('## '):
                section = line[3:].strip()
                continue
            for digits in DOC_ADDR_RE.findall(line):
                func = functions_by_address.get(int(digits, 16))
                if func is None or func.kind != 'game':
                    continue
                label = '%s: %s' % (rel_name, section)
                if label not in func.docs:
                    func.docs.append(label)


def read_coverage(coverage_dir_path, functions_by_address):
    """
    The traces of tools.func_coverage: which function ran in which phase of a scenario. A function is periodic
    in a phase when it ran in at least half of the windows of a phase of PERIODIC_MIN_WINDOWS windows or more.
    Returns the phases: [(scenario, phase, seconds, windows)].
    """
    phases = []
    if not coverage_dir_path or not os.path.isdir(coverage_dir_path):
        return phases
    file_names = sorted(n for n in os.listdir(coverage_dir_path) if n.endswith('.txt'))
    for file_name in file_names:
        scenario = file_name[:-len('.txt')]
        data = func_coverage.load(os.path.join(coverage_dir_path, file_name))
        windows = {}
        for index, (_, start_ms, name) in enumerate(data['phases']):
            end_ms = data['phases'][index + 1][1] if index + 1 < len(data['phases']) else data['end_ms']
            windows[name] = (end_ms - start_ms) / float(data['window_ms'])
            phases.append((scenario, name, (end_ms - start_ms) / 1000.0, windows[name]))
        for address, rows in data['hits'].items():
            func = functions_by_address.get(address)
            if func is None:
                continue
            for phase, first_ms, count in rows:
                # the phases of the only trace go without the name of its scenario
                key = phase if len(file_names) == 1 else '%s:%s' % (scenario, phase)
                func.coverage[key] = max(func.coverage.get(key, 0), count)
                if windows[phase] >= PERIODIC_MIN_WINDOWS and count >= windows[phase] / 2.0 and key not in func.periodic:
                    func.periodic.append(key)
    return phases


def resolve_status(functions, functions_by_address, decisions):
    for func in functions:
        decision = decisions.get('%06x' % func.address)
        if func.kind != 'game':
            func.status = func.kind
        elif any(p['symbol'] != '<module>' and not p.get('constant') and not p.get('partly') for p in func.python):
            func.status = 'implemented'
        elif any(p['symbol'] != '<module>' and not p.get('constant') for p in func.python):
            func.status = 'partial'
        elif func.python:
            func.status = 'module'
        elif decision:
            func.status = decision['decision']
            func.decision_note = decision.get('note')
        elif func.pattern:
            func.status, func.decision_note = func.pattern
    for func in functions:
        for address in func.calls:
            callee = functions_by_address.get(address)
            if callee is not None:
                callee.callers.add(func.address)
    for func in functions:
        if func.status is not None:
            continue
        covered = sorted(a for a in func.callers
                         if functions_by_address[a].status in ('implemented', 'partial', 'module'))
        if covered:
            func.status = 'indirect'
            func.covered_by = covered
        else:
            func.status = 'missing'


def guess_classes(functions):
    # one .cpp file is compiled into one block of code: an unnamed function between two methods of one class
    # most probably belongs to that class too
    game = [f for f in functions if f.kind == 'game']
    prev_class = None
    before = []
    for func in game:
        before.append(prev_class)
        if func.class_name and not func.class_name.startswith('shared_'):
            prev_class = func.class_name
    next_class = None
    for i in range(len(game) - 1, -1, -1):
        func = game[i]
        if not func.class_name and func.name.startswith('FUN_') and before[i] and before[i] == next_class:
            func.guess_class = next_class
        if func.class_name and not func.class_name.startswith('shared_'):
            next_class = func.class_name


def python_index(functions):
    index = collections.OrderedDict()
    for func in functions:
        for p in func.python:
            symbols = index.setdefault(p['file'], collections.OrderedDict())
            addresses = symbols.setdefault(p['symbol'], [])
            if '%06x' % func.address not in addresses:
                addresses.append('%06x' % func.address)
    return collections.OrderedDict(sorted(index.items()))


def write_json(json_file_path, c_rel_path, functions, py_index, unresolved, py_files):
    counts = collections.Counter(f.status for f in functions)
    game = [f for f in functions if f.kind == 'game']
    summary = collections.OrderedDict()
    summary['functions'] = len(functions)
    summary['game_functions'] = len(game)
    for status in STATUS_ORDER:
        summary[status] = counts.get(status, 0)
    summary['python_files'] = len(py_files)
    summary['python_files_with_refs'] = len(py_index)
    head = collections.OrderedDict()
    head['source'] = c_rel_path
    head['generator'] = 'PYTHONPATH=src venv/bin/python -m tools.func_registry'
    head['statuses'] = {
        'implemented': 'referenced from a function or a class of the remake',
        'partial': 'referenced only as "<address> (partly)": the remake does a part of the function',
        'builtin': 're/func_decisions.json: a container or an accessor whose job a Python built-in type does',
        'later': 're/func_decisions.json: needed, but not for the demo',
        'skip': 're/func_decisions.json: not needed by the remake',
        'module': 'referenced only by a module docstring or a module constant of a .py file',
        'indirect': 'not referenced, called directly by an implemented or module function (covered_by)',
        'missing': 'game code without any reference in remake/src',
        'library': 'CRT, MFC and imports of DLLs',
        'compiler': 'exception handlers made by the compiler (Unwind@, Catch@)',
    }
    head['summary'] = summary
    # one function per line: the file stays readable and diffs stay small
    parts = ['{\n']
    for key, value in head.items():
        parts.append(' %s: %s,\n' % (json.dumps(key), json.dumps(value, ensure_ascii=False)))
    parts.append(' "unresolved_python_refs": %s,\n' % json.dumps(
        [{'file': f, 'line': l, 'address': a} for f, l, a in unresolved]))
    parts.append(' "python_index": {\n')
    items = list(py_index.items())
    for i, (file_name, symbols) in enumerate(items):
        parts.append('  %s: %s%s\n' % (json.dumps(file_name), json.dumps(symbols), ',' if i + 1 < len(items) else ''))
    parts.append(' },\n')
    parts.append(' "functions": [\n')
    for i, func in enumerate(functions):
        parts.append('  %s%s\n' % (json.dumps(func.to_json(), ensure_ascii=False), ',' if i + 1 < len(functions) else ''))
    parts.append(' ]\n}\n')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))
    return summary


def decided(counts):
    return sum(counts[status] for status in DECISIONS)


def md_cell(text):
    return text.replace('|', '\\|')


def md_refs(func, limit=4):
    refs = []
    for p in func.python[:limit]:
        refs.append('`%s` %s:%d' % (p['symbol'], p['file'][len('src/'):], p['line']))
    if len(func.python) > limit:
        refs.append('+%d' % (len(func.python) - limit))
    return md_cell(', '.join(refs))


def write_markdown(md_file_path, json_rel_path, c_rel_path, functions, summary, py_index, py_rel_paths,
                   phases=None):
    game = [f for f in functions if f.kind == 'game']
    size_by_status = collections.Counter()
    for func in game:
        size_by_status[func.status] += func.size
    game_size = sum(size_by_status.values()) or 1
    out = []
    w = out.append
    w('# Реестр функций game.exe и их реализации в ремейке')
    w('')
    w('Каждая функция декомпилированного `%s` (Ghidra, первые «Проклятые Земли») сопоставлена с файлом и' % c_rel_path)
    w('функцией/классом в `remake/src`, которые её реализуют. Полный реестр (все %d функций) лежит в' % len(functions))
    w('[`%s`](%s), этот документ — его сводка.' % (json_rel_path, json_rel_path))
    w('')
    w('Оба файла пишет генератор, руками их не правят:')
    w('')
    w('```bash')
    w('make registry        # PYTHONPATH=src venv/bin/python -m tools.func_registry')
    w('```')
    w('')
    w('## Как находится соответствие')
    w('')
    w('В .py ремейка в комментариях и докстрингах стоят ссылки на код игры: адрес (`45ea20`, `0x50aaf0`,')
    w('`FUN_00461cb0`) или имя из `re/names.json` (`CUnitPrototype::Read`). Генератор читает каждый .py через')
    w('`tokenize` (только комментарии и строки, код не смотрится) и `ast`, и отдаёт ссылку той функции или классу,')
    w('внутри которых она стоит. Блок комментариев прямо над `def`, `class` или константой модуля относится к ним.')
    w('Адрес внутри тела функции (не её начало) относится к содержащей функции (`"inside": true`). Число без')
    w('шестнадцатеричных букв считается адресом, только если это точное начало функции.')
    w('')
    w('Статусы:')
    w('')
    w('| Статус | Значение |')
    w('|---|---|')
    w('| `implemented` | есть ссылка из функции или класса ремейка: реализация найдена |')
    w('| `partial` | ссылки только вида `5641f0 (partly)`: ремейк делает лишь часть функции |')
    w('| `module` | файл найден, но ссылка только в докстринге модуля или у константы (таблица, число из функции) |')
    w('| `builtin` | `re/func_decisions.json`: контейнер или доступ к полю, работу делает встроенный тип Python |')
    w('| `later` | `re/func_decisions.json`: нужно, но не для demo |')
    w('| `skip` | `re/func_decisions.json`: ремейку не нужно |')
    w('| `indirect` | ссылки нет, но функцию прямо вызывает реализованная (`covered_by`): возможно, она вошла в ту же реализацию |')
    w('| `missing` | игровой код, ссылок нет: **.py файл не найден** |')
    w('| `library` | CRT, MFC, импорты DLL (адреса от `%06x`) — реализовывать не нужно |' % LIBRARY_START)
    w('| `compiler` | обработчики исключений компилятора (`Unwind@`, `Catch@`) — не нужно |')
    w('')
    w('Ограничения:')
    w('')
    w('- соответствие найдено только там, где ремейк сам назвал адрес; код, написанный по поведению игры без ссылки,')
    w('  здесь числится `missing`;')
    w('- `indirect` — подсказка, а не проверка: вызванная функция могла и не понадобиться ремейку;')
    w('- `guess_class` у безымянной `FUN_` — класс её соседей по адресу (функции одного .cpp лежат подряд);')
    w('- имя (из `re/names.json` или Ghidra) есть только у %d игровых функций, остальные — `FUN_xxxxxxxx`.'
      % sum(1 for f in game if f.class_name or not f.name.startswith('FUN_')))
    w('')
    w('## Итог')
    w('')
    w('| Статус | Функций | Доля игрового кода по байтам |')
    w('|---|---:|---:|')
    for status in STATUS_ORDER:
        share = ''
        if status in size_by_status:
            share = '%.1f%%' % (100.0 * size_by_status[status] / game_size)
        w('| `%s` | %d | %s |' % (status, summary[status], share))
    w('| всего | %d | |' % summary['functions'])
    w('')
    w('Игровых функций (без `library` и `compiler`): %d. Python-файлов в `remake/src`: %d, из них ссылаются на'
      % (summary['game_functions'], summary['python_files']))
    w('game.exe %d.' % summary['python_files_with_refs'])
    w('')

    if phases:
        write_coverage(w, game, phases)

    # by domain
    w('## Покрытие по подсистемам')
    w('')
    w('Подсистема (`domain`) угадана `tools/func_domains.py`: по классу, строкам, вызовам DLL и .py реализации,')
    w('безымянным функциям она передаётся от соседей по адресу и по вызовам (`domain_source`). Уровень (`tier`):')
    w('`backend` заменён Python/Kivy/SDL и не переносится, `data` — форматы файлов (eilib), `game` — поведение игры.')
    w('')
    w('Колонка «решено» — `builtin`, `skip` и `later`.')
    w('')
    w('| Подсистема | Уровень | Всего | implemented | partial | module | решено | indirect | missing | Строк C в missing |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    by_domain = collections.defaultdict(list)
    for func in game:
        by_domain[func.domain].append(func)
    tier_order = {'game': 0, 'data': 1, 'backend': 2}
    for domain in sorted(by_domain, key=lambda d: (tier_order[func_domains.DOMAINS[d][0]], -len(by_domain[d]))):
        funcs = by_domain[domain]
        counts = collections.Counter(f.status for f in funcs)
        missing_lines = sum(f.lines for f in funcs if f.status == 'missing')
        w('| %s | %s | %d | %d | %d | %d | %d | %d | %d | %d |' % (domain, func_domains.DOMAINS[domain][0], len(funcs),
          counts['implemented'], counts['partial'], counts['module'], decided(counts), counts['indirect'], counts['missing'],
          missing_lines))
    w('')

    # by class
    w('## Покрытие по классам')
    w('')
    w('Классы из `re/names.json`; безымянные функции посчитаны по `guess_class`, остальные — в строке «без класса».')
    w('')
    w('| Класс | Всего | implemented | partial | module | решено | indirect | missing |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|')
    by_class = collections.defaultdict(collections.Counter)
    for func in game:
        by_class[func.class_name or func.guess_class or '(без класса)'][func.status] += 1
    for class_name, counts in sorted(by_class.items(), key=lambda kv: (kv[0].startswith('('), -sum(kv[1].values()), kv[0])):
        w('| %s | %d | %d | %d | %d | %d | %d | %d |' % (md_cell(class_name), sum(counts.values()),
          counts['implemented'], counts['partial'], counts['module'], decided(counts), counts['indirect'], counts['missing']))
    w('')

    # by address block
    w('## Покрытие по блокам адресов (64 КБ)')
    w('')
    w('| Блок | Всего | implemented | partial | module | решено | indirect | missing | Классы блока |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    blocks = collections.defaultdict(list)
    for func in game:
        blocks[func.address >> 16].append(func)
    for block in sorted(blocks):
        funcs = blocks[block]
        counts = collections.Counter(f.status for f in funcs)
        classes = collections.Counter(f.class_name for f in funcs if f.class_name)
        top = ', '.join(c for c, _ in classes.most_common(5))
        w('| %06x | %d | %d | %d | %d | %d | %d | %d | %s |' % (block << 16, len(funcs), counts['implemented'],
          counts['partial'], counts['module'], decided(counts), counts['indirect'], counts['missing'], md_cell(top)))
    w('')

    # python files
    w('## Python-файлы и функции game.exe')
    w('')
    w('| Файл | Символов со ссылками | Функций game.exe |')
    w('|---|---:|---:|')
    for file_name, symbols in py_index.items():
        addresses = set()
        for values in symbols.values():
            addresses.update(values)
        w('| `%s` | %d | %d |' % (file_name[len('src/'):], len(symbols), len(addresses)))
    w('')
    without = [p for p in py_rel_paths if p not in py_index]
    w('Без единой ссылки на game.exe (%d): %s.' % (len(without), ', '.join('`%s`' % p[len('src/'):] for p in without)))
    w('')

    # implemented
    for status, title in (('implemented', 'Реализованные функции (`implemented`)'),
                          ('partial', 'Частично перенесённые функции (`partial`)'),
                          ('module', 'Функции, найденные на уровне модуля (`module`)')):
        rows = [f for f in game if f.status == status]
        w('## %s — %d' % (title, len(rows)))
        w('')
        w('| Адрес | Функция C | Строк | Python |')
        w('|---|---|---:|---|')
        for func in rows:
            w('| %06x | %s | %d | %s |' % (func.address, md_cell(func.title()), func.lines, md_refs(func)))
        w('')

    # documented but not implemented
    rows = [f for f in game if f.docs and f.status in ('indirect', 'missing')]
    w('## Описаны в first_scan.md, но без реализации — %d' % len(rows))
    w('')
    w('| Адрес | Функция C | Статус | Строк | Раздел first_scan.md |')
    w('|---|---|---|---:|---|')
    for func in rows:
        sections = '; '.join(d.split(': ', 1)[1] for d in func.docs[:2])
        w('| %06x | %s | %s | %d | %s |' % (func.address, md_cell(func.title()), func.status, func.lines, md_cell(sections)))
    w('')

    # named but missing
    named = collections.defaultdict(list)
    for func in game:
        if func.status == 'missing' and (func.class_name or not func.name.startswith('FUN_')):
            named[func.class_name or '(свободные функции)'].append(func)
    w('## Именованные функции без реализации — %d' % sum(len(v) for v in named.values()))
    w('')
    w('Функции с именем из `re/names.json`, на которые ремейк не ссылается. Безымянные `missing` (%d) — только в JSON.'
      % sum(1 for f in game if f.status == 'missing' and not f.class_name and f.name.startswith('FUN_')))
    w('')
    w('| Класс | Методы (адрес) |')
    w('|---|---|')
    for class_name in sorted(named):
        methods = ', '.join('%s %06x' % (f.method_name or f.name, f.address) for f in named[class_name])
        w('| %s | %s |' % (md_cell(class_name), md_cell(methods)))
    w('')

    # json format
    w('## Формат JSON')
    w('')
    w('```')
    w('{')
    w('  "source", "generator", "statuses", "summary",')
    w('  "unresolved_python_refs": [{"file", "line", "address"}],   адреса из .py, которых нет в game_exe.c')
    w('  "python_index": {"src/x.py": {"Symbol.method": ["45ea20", ...]}},')
    w('  "functions": [   по возрастанию адреса, одна функция на строку')
    w('    {"address": "45ea20", "name": "FUN_0045ea20", "status": "implemented",')
    w('     "lines": 120, "size": 1424,            строк C и байт кода до следующей функции')
    w('     "class", "method" | "guess_class",')
    w('     "domain", "domain_source", "tier",     подсистема, откуда она взята, backend/data/game')
    w('     "callers": 3, "calls": 12,             прямые вызовы внутри game.exe')
    w('     "python": [{"file", "symbol", "line", "via": "address"|"name", "constant"?, "inside"?}],')
    w('     "covered_by": ["..."],                 для indirect: реализованные вызывающие функции')
    w('     "docs": ["first_scan.md: раздел"],')
    w('     "strings": ["..."]}                    строки, которые использует функция (подсказка, что она делает)')
    w('  ]')
    w('}')
    w('```')
    w('')
    w('У `library` и `compiler` записаны только адрес, имя, статус (и размер у `library`).')
    w('')
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


def write_coverage(w, game, phases):
    traced = [f for f in game if f.coverage]
    periodic = [f for f in traced if f.periodic]
    w('## Трасса оригинала (`re/coverage`)')
    w('')
    w('`tools/func_coverage.py` запускает game.exe под отладчиком `re/eitrace` (int3 на первом байте каждой игровой')
    w('функции) и проходит сценарий `eitools/scenarios/<имя>.txt`. Сценарий делится на фазы; в каждой фазе все')
    w('функции заново ставятся на int3, а сработавшие — ещё раз в конце каждого окна в 2 с. Функция')
    w('**периодическая** в фазе, если работала хотя бы в половине окон фазы (фаза от 5 окон, 10 с): это код')
    w('каждого тика.')
    w('В JSON: `coverage` — {фаза: окон}, `periodic` — фазы, где функция периодическая.')
    w('')
    w('| Сценарий | Фаза | Секунд | Функций | Периодических |')
    w('|---|---|---:|---:|---:|')
    for scenario, phase, seconds, windows in phases:
        key = phase if len(set(p[0] for p in phases)) == 1 else '%s:%s' % (scenario, phase)
        count = sum(1 for f in traced if key in f.coverage)
        count_periodic = sum(1 for f in traced if key in f.periodic)
        w('| %s | %s | %.1f | %d | %d |' % (scenario, phase, seconds, count, count_periodic))
    w('')
    w('| Статус | Сработали в трассе | Из них периодические |')
    w('|---|---:|---:|')
    for status in STATUS_ORDER:
        rows = [f for f in traced if f.status == status]
        if rows:
            w('| `%s` | %d | %d |' % (status, len(rows), sum(1 for f in rows if f.periodic)))
    w('| всего | %d | %d |' % (len(traced), len(periodic)))
    w('')
    w('Игровых функций, которые в трассе ни разу не сработали: %d.' % (len(game) - len(traced)))
    w('')
    todo = [f for f in traced if f.status in ('missing', 'indirect')]
    w('### Сработали, но не реализованы, по подсистемам')
    w('')
    w('| Подсистема | Уровень | missing | indirect | периодические | Строк C |')
    w('|---|---|---:|---:|---:|---:|')
    by_domain = collections.defaultdict(list)
    for func in todo:
        by_domain[func.domain].append(func)
    for domain in sorted(by_domain, key=lambda d: -len(by_domain[d])):
        rows = by_domain[domain]
        w('| %s | %s | %d | %d | %d | %d |' % (domain, func_domains.DOMAINS[domain][0],
          sum(1 for f in rows if f.status == 'missing'), sum(1 for f in rows if f.status == 'indirect'),
          sum(1 for f in rows if f.periodic), sum(f.lines for f in rows)))
    w('')


def read_decisions(decisions_file_path):
    if not decisions_file_path or not os.path.isfile(decisions_file_path):
        return {}
    with open(decisions_file_path, encoding='utf-8') as f:
        decisions = json.load(f)
    for address, decision in decisions.items():
        if decision.get('decision') not in DECISIONS:
            raise ValueError('%s: unknown decision %r' % (address, decision.get('decision')))
    return decisions


def build(c_file_path, names_file_path, src_dir_path, doc_file_path, json_file_path, md_file_path,
          decisions_file_path=None, coverage_dir_path=None):
    with open(names_file_path, encoding='utf-8') as f:
        names = json.load(f)
    decisions = read_decisions(decisions_file_path)
    functions = read_c(c_file_path, names)
    functions_by_address = {f.address: f for f in functions}
    starts = sorted(functions_by_address)
    names_by_full = {}
    for address, full_name in names.items():
        names_by_full[full_name.rstrip('?')] = int(address, 16)
    py_files = []
    for dir_path, dir_names, file_names in os.walk(src_dir_path):
        dir_names[:] = sorted(d for d in dir_names if d != '__pycache__')
        for file_name in sorted(file_names):
            # the registry tools name addresses only as examples
            if file_name.endswith('.py') and not file_name.startswith('func_'):
                py_files.append(os.path.join(dir_path, file_name))
    unresolved = []
    for py_file_path in py_files:
        rel_path = os.path.relpath(py_file_path, REMAKE_PATH)
        refs, missed = read_python(py_file_path, rel_path, functions_by_address, starts, names_by_full)
        unresolved.extend(missed)
        for func, item in refs:
            key = (item['file'], item['symbol'])
            if any((p['file'], p['symbol']) == key for p in func.python):
                continue
            func.python.append(item)
    if doc_file_path and os.path.isfile(doc_file_path):
        read_docs(doc_file_path, functions_by_address)
    resolve_status(functions, functions_by_address, decisions)
    phases = read_coverage(coverage_dir_path, functions_by_address)
    guess_classes(functions)
    func_domains.assign(functions)
    py_index = python_index(functions)
    c_rel_path = os.path.relpath(c_file_path, REMAKE_PATH)
    summary = write_json(json_file_path, c_rel_path, functions, py_index, unresolved, py_files)
    if md_file_path:
        json_rel_path = os.path.relpath(json_file_path, os.path.dirname(os.path.abspath(md_file_path)))
        py_rel_paths = [os.path.relpath(p, REMAKE_PATH) for p in py_files]
        write_markdown(md_file_path, json_rel_path, c_rel_path, functions, summary, py_index, py_rel_paths,
                       phases)
    if _Debug:
        print(json.dumps(summary, indent=1))
    return functions, summary


def main():
    parser = argparse.ArgumentParser(description='registry of the functions of game.exe and their implementations')
    parser.add_argument('--c', dest='c_file_path', metavar='C_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 'build', 'ghidra', 'game_exe.c'))
    parser.add_argument('--names', dest='names_file_path', metavar='NAMES_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 're', 'names.json'))
    parser.add_argument('--src', dest='src_dir_path', metavar='SRC_DIR_PATH',
                        default=os.path.join(REMAKE_PATH, 'src'))
    parser.add_argument('--doc', dest='doc_file_path', metavar='DOC_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 'first_scan.md'))
    parser.add_argument('-o', dest='json_file_path', metavar='JSON_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 'func_registry.json'))
    parser.add_argument('--decisions', dest='decisions_file_path', metavar='DECISIONS_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 're', 'func_decisions.json'))
    parser.add_argument('--coverage', dest='coverage_dir_path', metavar='COVERAGE_DIR_PATH',
                        default=os.path.join(REMAKE_PATH, 're', 'coverage'))
    parser.add_argument('--md', dest='md_file_path', metavar='MD_FILE_PATH',
                        default=os.path.join(REMAKE_PATH, 'FUNC_REGISTRY.md'))
    args = parser.parse_args()
    build(args.c_file_path, args.names_file_path, args.src_dir_path, args.doc_file_path, args.json_file_path,
          args.md_file_path, args.decisions_file_path, args.coverage_dir_path)


if __name__ == '__main__':
    main()
