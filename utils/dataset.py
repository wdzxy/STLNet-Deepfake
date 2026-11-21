import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class Record:
    path: Path
    label: int
    video_id: str = ""
    source_id: str = ""


def read_manifest(filename):
    """Relative paths are relative to the CSV, not the current directory."""
    filename = Path(filename).resolve()
    records = []
    seen = set()
    video_labels = {}
    with filename.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{filename}: CSV requires path,label columns")
        for line, row in enumerate(reader, 2):
            raw_path = (row.get("path") or "").strip()
            if not raw_path or (row.get("label") or "").strip() not in {"0", "1"}:
                raise ValueError(f"{filename}:{line}: expected nonempty path and label 0 or 1")
            path = Path(raw_path)
            path = (filename.parent / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{filename}:{line}: {path}")
            if path in seen:
                raise ValueError(f"Duplicate image in manifest: {path}")
            seen.add(path)
            label = int(row["label"])
            video = (row.get("video_id") or "").strip()
            source = (row.get("source_id") or "").strip()
            if video and video in video_labels and video_labels[video] != label:
                raise ValueError(f"video_id {video!r} has conflicting labels")
            if video:
                video_labels[video] = label
            records.append(Record(path, label, video, source))
    if not records:
        raise ValueError(f"Empty manifest: {filename}")
    return records


def check_split_leakage(train_records, val_records):
    """Require disjoint known image, video and original-source identities."""
    for attribute in ("path", "video_id", "source_id"):
        left = {getattr(r, attribute) for r in train_records if getattr(r, attribute)}
        right = {getattr(r, attribute) for r in val_records if getattr(r, attribute)}
        overlap = left & right
        if overlap:
            raise ValueError(f"Train/validation leakage in {attribute}: {list(map(str, overlap))[:5]}")
    for name, records in (("train", train_records), ("validation", val_records)):
        if {r.label for r in records} != {0, 1}:
            raise ValueError(f"{name} must contain both real (0) and forged (1) images")
    missing = [name for name, records in (("train", train_records), ("validation", val_records))
               if any(not r.source_id for r in records)]
    if missing:
        print("WARNING: source_id is missing for some " + ", ".join(missing)
              + " samples; original-source/identity leakage cannot be fully checked.", flush=True)


class FaceTransform:

    def __init__(self, config, training=False):
        self.size = int(config.get("image_size", 224))
        if self.size < 16:
            raise ValueError("image_size must be at least 16")
        self.mean = np.asarray(config.get("mean", [0.5] * 3), dtype=np.float32)
        self.std = np.asarray(config.get("std", [0.5] * 3), dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
            raise ValueError("mean/std must have three components and std must be positive")
        self.training = training
        self.augment = config.get("augmentation", {})
        low = float(self.augment.get("crop_scale_min", 0.9))
        if not 0 < low <= 1:
            raise ValueError("crop_scale_min must lie in (0, 1]")
        for key in ("rotation_degrees", "brightness", "noise_std"):
            if float(self.augment.get(key, 0)) < 0:
                raise ValueError(f"{key} must be nonnegative")

    def __call__(self, image):
        image = image.convert("RGB")
        if self.training:
            scale = random.uniform(float(self.augment.get("crop_scale_min", 0.9)), 1.0)
            width, height = image.size
            edge = max(1, round(min(width, height) * math.sqrt(scale)))
            left = random.randint(0, width - edge)
            top = random.randint(0, height - edge)
            image = image.crop((left, top, left + edge, top + edge))
            rotation = float(self.augment.get("rotation_degrees", 10.0))
            image = image.rotate(random.uniform(-rotation, rotation), resample=Image.Resampling.BILINEAR)
            brightness = float(self.augment.get("brightness", 0.1))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(max(0, 1-brightness), 1+brightness))
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.training:
            sigma = float(self.augment.get("noise_std", 0.01))
            if sigma:
                array = np.clip(array + np.random.normal(0, sigma, array.shape).astype(np.float32), 0, 1)
        array = (array - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class FaceDataset(Dataset):
    def __init__(self, records, config, training=False):
        self.records = records
        self.transform = FaceTransform(config, training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor(record.label, dtype=torch.float32), index


class NoSingletonBatchSampler(Sampler):

    def __init__(self, count, batch_size, seed=42):
        if count < 2 or batch_size < 2:
            raise ValueError("Training requires at least two images and micro_batch_size >= 2")
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.epoch = 0

    def __iter__(self):
        indices = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        batches = [indices[i:i+self.batch_size] for i in range(0, self.count, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        result = math.ceil(self.count / self.batch_size)
        return result - int(result > 1 and self.count % self.batch_size == 1)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
