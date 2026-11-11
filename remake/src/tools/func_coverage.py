"""
Runs the original game.exe under the coverage tracer re/eitrace and keeps which functions ran in which phase
of a scenario of eitools/scenarios.

    PYTHONPATH=src venv/bin/python -m tools.func_coverage run coverage        build, run, collect
    PYTHONPATH=src venv/bin/python -m tools.func_coverage collect coverage    only take eitrace_out.txt
    PYTHONPATH=src venv/bin/python -m tools.func_coverage show coverage       phases and counts

The tracer lives in the Wine prefix of the game (C:\\eitrace, there is no Z: drive), the game is started by
EvilIslands1/trace.sh (a copy of start.sh which reads game_info/game_info_trace.txt). The result goes to
re/coverage/<scenario>.txt in the format of eitrace (P and H lines), read by tools.func_registry.
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import os
import sys
import json
import time
import shutil
import argparse
import subprocess

#------------------------------------------------------------------------------

REMAKE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HOME_PATH = os.path.dirname(REMAKE_PATH)
GAME_PATH = os.path.join(HOME_PATH, 'EvilIslands1')
TRACER_DIR_PATH = os.path.join(GAME_PATH, 'prefix', 'drive_c', 'eitrace')
EIDRIVE_PATH = os.path.join(REMAKE_PATH, 'eitools', 'eidrive.py')
SCENARIOS_PATH = os.path.join(REMAKE_PATH, 'eitools', 'scenarios')
COVERAGE_PATH = os.path.join(REMAKE_PATH, 're', 'coverage')
SOURCE_FILE_PATH = os.path.join(REMAKE_PATH, 're', 'eitrace', 'eitrace.c')
EXE_FILE_PATH = os.path.join(REMAKE_PATH, 'build', 'eitrace', 'eitrace.exe')

# the statuses whose functions get no int3: code of the C runtime, MFC and the compiler
UNTRACED = ('library', 'compiler')

GAME_INFO_TRACE = 'EvilIslands\n1.0.6\nC:\\eitrace\\eitrace.exe\nK:\\game_info\\data\n\nEvil Islands\n'

#------------------------------------------------------------------------------


def build_tracer():
    if os.path.exists(EXE_FILE_PATH) and os.path.getmtime(EXE_FILE_PATH) >= os.path.getmtime(SOURCE_FILE_PATH):
        return
    os.makedirs(os.path.dirname(EXE_FILE_PATH), exist_ok=True)
    subprocess.check_call(['i686-w64-mingw32-gcc', '-O2', '-Wall', '-o', EXE_FILE_PATH, SOURCE_FILE_PATH])


def install_tracer():
    """
    eitrace.exe and the list of addresses into C:\\eitrace of the prefix, the launcher trace.sh next to
    start.sh of the game.
    """
    build_tracer()
    os.makedirs(TRACER_DIR_PATH, exist_ok=True)
    shutil.copy(EXE_FILE_PATH, os.path.join(TRACER_DIR_PATH, 'eitrace.exe'))
    with open(os.path.join(REMAKE_PATH, 'func_registry.json'), encoding='utf-8') as f:
        registry = json.load(f)
    addresses = [e['address'] for e in registry['functions'] if e['status'] not in UNTRACED]
    with open(os.path.join(TRACER_DIR_PATH, 'eitrace_addrs.txt'), 'w') as f:
        f.write('\n'.join(addresses) + '\n')
    for name in ('eitrace_out.txt', 'eitrace.log'):
        file_path = os.path.join(TRACER_DIR_PATH, name)
        if os.path.exists(file_path):
            os.remove(file_path)
    with open(os.path.join(TRACER_DIR_PATH, 'eitrace_phase.txt'), 'w') as f:
        f.write('boot\n')
    trace_sh_path = os.path.join(GAME_PATH, 'trace.sh')
    if not os.path.exists(trace_sh_path):
        shutil.copy(os.path.join(GAME_PATH, 'start.sh'), trace_sh_path)
    settings_path = os.path.join(GAME_PATH, 'settings_trace')
    if not os.path.exists(settings_path):
        shutil.copy(os.path.join(GAME_PATH, 'settings_start'), settings_path)
    with open(os.path.join(GAME_PATH, 'game_info', 'game_info_trace.txt'), 'w') as f:
        f.write(GAME_INFO_TRACE)
    if _Debug:
        print('installed the tracer with %d addresses' % len(addresses))
    return len(addresses)


def eidrive(*args, **kwargs):
    env = dict(os.environ, EIDRIVE_HOME=HOME_PATH,
               EIDRIVE_PHASE_FILE=os.path.join(TRACER_DIR_PATH, 'eitrace_phase.txt'))
    env.update(kwargs.get('env', {}))
    return subprocess.call([sys.executable, EIDRIVE_PATH, 'original'] + list(args), env=env)


def wait_tracer_done(timeout):
    log_file_path = os.path.join(TRACER_DIR_PATH, 'eitrace.log')
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_file_path):
            with open(log_file_path, encoding='latin-1') as f:
                if f.read().rstrip().endswith('done'):
                    return True
        time.sleep(1)
    return False


def collect(scenario):
    out_file_path = os.path.join(TRACER_DIR_PATH, 'eitrace_out.txt')
    if not os.path.exists(out_file_path):
        raise SystemExit('no %s' % out_file_path)
    os.makedirs(COVERAGE_PATH, exist_ok=True)
    dest_file_path = os.path.join(COVERAGE_PATH, scenario + '.txt')
    shutil.copy(out_file_path, dest_file_path)
    return dest_file_path


def run(scenario):
    scenario_file_path = os.path.join(SCENARIOS_PATH, scenario + '.txt')
    if not os.path.exists(scenario_file_path):
        raise SystemExit('no scenario %s' % scenario_file_path)
    eidrive('stop')
    count = install_tracer()
    print('tracer installed, %d functions' % count)
    if eidrive('start', env={'EIDRIVE_ORIGINAL_SCRIPT': 'trace.sh'}) != 0:
        raise SystemExit('the game did not start')
    code = eidrive('run', scenario_file_path)
    if not wait_tracer_done(30):
        print('the game did not exit, stopping it')
        eidrive('stop')
        time.sleep(2)
    dest_file_path = collect(scenario)
    print('scenario exit code %d, coverage %s' % (code, dest_file_path))
    show(scenario)


def load(coverage_file_path):
    """
    {'phases': [(index, start_ms, name)], 'hits': {address: [(phase name, first_ms, windows)]},
     'window_ms': the length of a window, 'end_ms': the time of the last write}
    """
    phases = {}
    hits = {}
    head = {}
    with open(coverage_file_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == '#':
                # "# eitrace window_ms 2000 now 20166 funcs 10555"
                head = dict(zip(parts[2::2], parts[3::2]))
                continue
            if parts[0] == 'P':
                phases[int(parts[1])] = (int(parts[1]), int(parts[2]), parts[3])
            elif parts[0] == 'H':
                address = int(parts[1], 16)
                hits.setdefault(address, []).append((phases[int(parts[2])][2], int(parts[3]), int(parts[4])))
    return {'phases': [phases[k] for k in sorted(phases)], 'hits': hits,
            'window_ms': int(head.get('window_ms', 2000)), 'end_ms': int(head.get('now', 0))}


def show(scenario):
    data = load(os.path.join(COVERAGE_PATH, scenario + '.txt'))
    counts = {}
    for address, rows in data['hits'].items():
        for phase, first_ms, windows in rows:
            counts[phase] = counts.get(phase, 0) + 1
    for index, start_ms, name in data['phases']:
        print('%-14s %8.1f s %6d functions' % (name, start_ms / 1000.0, counts.get(name, 0)))
    print('%-14s %17d functions' % ('all', len(data['hits'])))


def main():
    parser = argparse.ArgumentParser(description='coverage of game.exe functions by scenario')
    parser.add_argument('command', choices=['run', 'collect', 'show', 'install'])
    parser.add_argument('scenario', nargs='?', default='coverage')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.scenario)
    elif args.command == 'collect':
        print(collect(args.scenario))
        show(args.scenario)
    elif args.command == 'show':
        show(args.scenario)
    else:
        print('installed, %d functions' % install_tracer())


if __name__ == '__main__':
    main()
