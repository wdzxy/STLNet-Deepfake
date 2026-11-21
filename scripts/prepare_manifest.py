import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Split a face manifest by source group')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must lie strictly between 0 and 1')
    source = Path(args.input).resolve()
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not {'path', 'label', 'source_id'}.issubset(fields or []):
            raise ValueError('Input must contain path,label,source_id columns')
        rows = list(reader)
    seen_paths, video_sources = (set(), {})
    for row in rows:
        if not (row.get('source_id') or '').strip() or row.get('label') not in {'0', '1'}:
            raise ValueError('Every row requires a source_id and binary label')
        path = Path(row.get('path') or '')
        path = (source.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in seen_paths:
            raise ValueError(f'Duplicate path: {path}')
        seen_paths.add(path)
        row['path'] = str(path)
        row['source_id'] = row['source_id'].strip()
        video = (row.get('video_id') or '').strip()
        if video:
            if video in video_sources and video_sources[video] != row['source_id']:
                raise ValueError(f'Video {video} maps to multiple source groups')
            video_sources[video] = row['source_id']
    groups = sorted({row['source_id'] for row in rows})
    if len(groups) < 2:
        raise ValueError('Need at least two independent source groups')
    random.Random(args.seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * args.val_fraction)))
    validation = set(groups[:count])
    split = {'train': [r for r in rows if r['source_id'] not in validation], 'val': [r for r in rows if r['source_id'] in validation]}
    for name, subset in split.items():
        if {row['label'] for row in subset} != {'0', '1'}:
            raise ValueError(f'{name} has only one class; supply a valid official split or another seed')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(((destination / f'{name}.csv').exists() for name in split)):
        raise FileExistsError('Split files already exist; choose a new output directory')
    for name, subset in split.items():
        with (destination / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
        print(f"{name}: {len(subset)} images, {len({r['source_id'] for r in subset})} source groups")
    print('The ratio is by source groups, not frames. CSV paths are absolute for portability across output locations.')
if __name__ == '__main__':
    main()
