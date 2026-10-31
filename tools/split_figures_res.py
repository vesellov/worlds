"""
Splits big figures.res into many small .res files: one file per model (template), named <template>.res.

The client downloads only the models listed by the game server instead of the whole figures.res,
every small file is read by the client with the same functions from src/res.py as figures.res before.

    venv/bin/python3 tools/split_figures_res.py ~/.worlds/data/figures.res ./figures/
    venv/bin/python3 tools/split_figures_res.py ~/.worlds/data/figures.res ./figures/ --templates unhuma,stbuho30
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import res  # @UnresolvedImport


def main():
    parser = argparse.ArgumentParser(description='split figures.res into one .res file per template')
    parser.add_argument('source', help='path to figures.res')
    parser.add_argument('destination', help='output folder for <template>.res files')
    parser.add_argument('--templates', default='', help='comma separated list of templates, all templates by default')
    args = parser.parse_args()
    if not os.path.isdir(args.destination):
        os.makedirs(args.destination)
    selected = set(t.strip().lower() for t in args.templates.split(',') if t.strip())
    with open(args.source, 'rb') as figures_file:
        filetree_dict = res.read_res_filetree(figures_file, return_dict=True)
        templates, not_assigned = res.figures_templates(figures_file, filetree_dict)
        if selected:
            unknown = selected - set(templates.keys())
            if unknown:
                print(f'unknown templates: {sorted(unknown)}')
                return 1
            templates = {t: templates[t] for t in selected}
        written = 0
        total_size = 0
        sizes = []
        errors = 0
        for t in sorted(templates.keys()):
            dest_file_path = os.path.join(args.destination, t + '.res')
            res.extract_figures_template(figures_file, filetree_dict, templates[t], dest_file_path)
            # read the new file back and compare every element with the source
            with open(dest_file_path, 'rb') as new_file:
                new_filetree = res.read_res_filetree(new_file, return_dict=True)
                if len(new_filetree) != len(templates[t]):
                    print(f'ERROR: {t} has {len(new_filetree)} elements, expected {len(templates[t])}')
                    errors += 1
                for name in templates[t]:
                    element = new_filetree.get(name.lower())
                    if not element:
                        print(f'ERROR: {name} is missing in {dest_file_path}')
                        errors += 1
                        continue
                    _, src_offset, src_size = filetree_dict[name]
                    figures_file.seek(src_offset)
                    src_data = figures_file.read(src_size)
                    new_file.seek(element[1])
                    if new_file.read(element[2]) != src_data:
                        print(f'ERROR: {name} content differs in {dest_file_path}')
                        errors += 1
            size = os.path.getsize(dest_file_path)
            sizes.append((size, t))
            total_size += size
            written += 1
    print(f'written {written} files into {args.destination}, total size {total_size / 1e6:.1f} MB')
    print('largest: ' + ', '.join(f'{t} {s / 1e6:.1f} MB' for s, t in sorted(sizes, reverse=True)[:10]))
    if not_assigned and not selected:
        print(f'{len(not_assigned)} elements do not belong to any template and were skipped: {sorted(not_assigned)}')
    if errors:
        print(f'{errors} errors found')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
