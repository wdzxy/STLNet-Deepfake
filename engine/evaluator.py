import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.dataset import FaceDataset, seed_worker
from utils.metrics import binary_metrics


def make_eval_loader(records, data_config, batch_size=16, workers=0):
    return DataLoader(FaceDataset(records, data_config), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)


@torch.no_grad()
def evaluate(model, loader, records, device, threshold=0.5, predictions_path=None, metadata=None):
    model.eval()
    probabilities = np.empty(len(records), dtype=np.float64)
    seen = np.zeros(len(records), dtype=bool)
    loss_sum = 0.0
    for images, labels, indices in loader:
        logits = model(images.to(device))
        loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device), reduction="sum").item()
        index_array = indices.numpy()
        if seen[index_array].any():
            raise ValueError("Evaluation loader emitted duplicate indices")
        seen[index_array] = True
        probabilities[index_array] = logits.sigmoid().cpu().numpy()
    if not seen.all():
        raise ValueError("Evaluation loader did not cover the entire manifest")
    labels = np.asarray([r.label for r in records])
    metrics = binary_metrics(labels, probabilities, threshold)
    result = {"frame_" + key: value for key, value in metrics.items()}
    result.update(frame_loss=loss_sum / len(records), frames=len(records), threshold=threshold,
                  video_accuracy=None, video_auc=None, video_precision=None, video_recall=None,
                  video_f1=None, videos=0)
    if all(r.video_id for r in records):
        groups = {}
        for record, probability in zip(records, probabilities):
            group = groups.setdefault(record.video_id, {"label": record.label, "scores": []})
            if group["label"] != record.label:
                raise ValueError(f"Conflicting labels for video {record.video_id}")
            group["scores"].append(float(probability))
        video_result = binary_metrics([v["label"] for v in groups.values()],
                                      [np.mean(v["scores"]) for v in groups.values()], threshold)
        result.update({"video_" + key: value for key, value in video_result.items()})
        result["videos"] = len(groups)
    if predictions_path is not None:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if metadata is not None and len(metadata) != len(records):
            raise ValueError("Prediction metadata must align with the manifest")
        for index, (record, probability) in enumerate(zip(records, probabilities)):
            row = dict(metadata[index]) if metadata is not None else {}
            row.update(path=str(record.path), label=record.label, video_id=record.video_id,
                       source_id=record.source_id, probability=float(probability))
            rows.append(row)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return result
