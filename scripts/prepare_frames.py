from __future__ import annotations
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import math
import re
from pathlib import Path
import numpy as np
from PIL import Image
_SAFE_ID = re.compile('[A-Za-z0-9][A-Za-z0-9_-]{0,127}\\Z')
_WINDOWS_RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{number}' for prefix in ('COM', 'LPT') for number in range(1, 10)}

def validate_video_id(value):
    if not _SAFE_ID.fullmatch(value) or value.upper() in _WINDOWS_RESERVED:
        raise ValueError(f'Unsafe video_id {value!r}: use 1-128 ASCII letters, digits, underscores or hyphens')
    return value

def uniform_frame_indices(total, count=30):
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError('Video frame count must be a positive integer')
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('frames_per_video must be a positive integer')
    return np.unique(np.linspace(0, total - 1, min(total, count), dtype=np.int64)).tolist()

def read_videos(filename):
    filename = Path(filename).resolve()
    records, ids, paths = ([], set(), set())
    with filename.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'path', 'label', 'video_id', 'source_id'}.issubset(reader.fieldnames or []):
            raise ValueError('Video CSV requires path,label,video_id,source_id columns')
        optional = [name for name in ('method', 'category') if name in reader.fieldnames]
        for line, row in enumerate(reader, 2):
            record = {name: (row.get(name) or '').strip() for name in reader.fieldnames}
            video_id = validate_video_id(record['video_id'])
            if video_id.casefold() in ids:
                raise ValueError(f'Duplicate video_id (case-insensitive): {video_id}')
            if record['label'] not in ('0', '1') or not record['source_id'] or (not record['path']):
                raise ValueError(f'{filename}:{line}: require label 0/1 and nonempty path/source_id')
            path = Path(record['path'])
            path = (filename.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            path_key = str(path).casefold()
            if path_key in paths:
                raise ValueError(f'Duplicate source video path: {path}')
            record['path'] = path
            records.append(record)
            ids.add(video_id.casefold())
            paths.add(path_key)
    if not records:
        raise ValueError('Video CSV is empty')
    return (records, optional)

def read_boxes(filename, video_ids):
    boxes = {}
    with Path(filename).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'video_id', 'frame_index', 'x1', 'y1', 'x2', 'y2'}.issubset(reader.fieldnames or []):
            raise ValueError('Boxes CSV requires video_id,frame_index,x1,y1,x2,y2 columns')
        for line, row in enumerate(reader, 2):
            video_id = (row.get('video_id') or '').strip()
            if video_id not in video_ids:
                raise ValueError(f'Boxes CSV:{line}: unknown video_id {video_id!r}')
            try:
                index = int(row['frame_index'])
                coords = tuple((float(row[name]) for name in ('x1', 'y1', 'x2', 'y2')))
            except (ValueError, TypeError) as error:
                raise ValueError(f'Boxes CSV:{line}: invalid frame index or coordinate') from error
            if index < 0 or not all((math.isfinite(value) for value in coords)):
                raise ValueError(f'Boxes CSV:{line}: index must be nonnegative and coordinates finite')
            x1, y1, x2, y2 = coords
            if x1 < 0 or y1 < 0 or x2 <= x1 or (y2 <= y1):
                raise ValueError(f'Boxes CSV:{line}: expected 0 <= x1 < x2 and 0 <= y1 < y2')
            key = (video_id, index)
            if key in boxes:
                raise ValueError(f'Duplicate face box for video/frame {key}')
            boxes[key] = coords
    return boxes

def prepare_frames(videos, output, boxes=None, already_face_cropped=False, frames_per_video=30):
    if (boxes is not None) == bool(already_face_cropped):
        raise ValueError('Choose exactly one of boxes or already_face_cropped=True')
    uniform_frame_indices(1, frames_per_video)
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Video preprocessing requires the optional dependency: pip install '.[video]'") from error
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output must be a new directory; refusing to overwrite: {output}')
    records, optional = read_videos(videos)
    face_boxes = read_boxes(boxes, {record['video_id'] for record in records}) if boxes is not None else {}
    plans = []
    for record in records:
        capture = cv2.VideoCapture(str(record['path']))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {record['path']}")
            raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(raw_count) or raw_count < 1:
                raise ValueError(f"Cannot determine frame count: {record['path']}")
            indices = uniform_frame_indices(int(raw_count), frames_per_video)
        finally:
            capture.release()
        if boxes is not None:
            missing = [index for index in indices if (record['video_id'], index) not in face_boxes]
            if missing:
                raise ValueError(f"Missing face boxes for {record['video_id']} at sampled frames {missing}")
        plans.append((record, indices))
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / 'INCOMPLETE'
    incomplete.write_text('Preprocessing has not completed. Do not use partial images as a finished dataset.\n', encoding='utf-8')
    manifest_rows = []
    for record, indices in plans:
        video_id = record['video_id']
        destination = output / 'frames' / video_id
        destination.mkdir(parents=True, exist_ok=False)
        selected = set(indices)
        capture = cv2.VideoCapture(str(record['path']))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot reopen video: {record['path']}")
            for index in range(indices[-1] + 1):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"Cannot decode {record['path']} at frame {index}")
                if index not in selected:
                    continue
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if boxes is not None:
                    x1, y1, x2, y2 = face_boxes[video_id, index]
                    if x2 > image.width or y2 > image.height:
                        raise ValueError(f'Face box outside frame bounds for {video_id}, frame {index}')
                    image = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
                image = image.resize((224, 224), Image.Resampling.BILINEAR)
                filename = destination / f'{index:06d}.png'
                with filename.open('xb') as handle:
                    image.save(handle, format='PNG')
                manifest_rows.append({'path': filename.relative_to(output).as_posix(), 'label': record['label'], 'video_id': video_id, 'source_id': record['source_id'], 'frame_index': index, **{name: record[name] for name in optional}})
        finally:
            capture.release()
    manifest = output / 'manifest.csv'
    with manifest.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'video_id', 'source_id', 'frame_index'] + optional)
        writer.writeheader()
        writer.writerows(manifest_rows)
    incomplete.unlink()
    return (manifest, len(manifest_rows))

def main(argv=None):
    parser = argparse.ArgumentParser(description='Extract uniformly sampled face crops')
    parser.add_argument('--videos', required=True, help='CSV: path,label,video_id,source_id[,method,category]')
    parser.add_argument('--output', required=True, help='New output directory; existing paths are refused')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--boxes', help='CSV: video_id,frame_index,x1,y1,x2,y2; zero-based frames, pixel xyxy')
    source.add_argument('--already-face-cropped', action='store_true', help='Explicitly confirm videos contain face crops')
    parser.add_argument('--frames-per-video', type=int, default=30, help='Uniform samples; default 30 follows HFF')
    args = parser.parse_args(argv)
    try:
        manifest, count = prepare_frames(args.videos, args.output, args.boxes, args.already_face_cropped, args.frames_per_video)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Preprocessing failed: {error}\n')
    print(f'Exported {count} RGB face crops at 224x224. Manifest: {manifest}')
if __name__ == '__main__':
    main()
