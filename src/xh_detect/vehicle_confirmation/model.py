from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


class VehicleConfirmer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        feature_dim = backbone.classifier[0].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.scalar_encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim + 16, 1))

    def forward(self, images: Tensor, scalar_features: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width)")
        if scalar_features.ndim != 2 or scalar_features.shape[1] != 3:
            raise ValueError("scalar_features must have shape (batch, 3)")
        if images.shape[0] != scalar_features.shape[0]:
            raise ValueError("images and scalar_features batch sizes must match")
        image_features = self.avgpool(self.features(images)).flatten(1)
        scalar_embedding = self.scalar_encoder(scalar_features)
        return self.classifier(torch.cat((image_features, scalar_embedding), dim=1)).squeeze(1)


@dataclass(frozen=True)
class VehicleConfirmerTrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 4
    seed: int = 42
    pretrained: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workers", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.pretrained, bool):
            raise TypeError("pretrained must be a boolean")


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"manifest does not exist: {path}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("label") not in {0, 1}:
            raise ValueError(f"manifest line {line_number} is invalid")
        records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


class _VehicleDataset(Dataset):
    def __init__(self, root: Path, manifest: Path) -> None:
        self.root = root
        self.records = _load_records(manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        image_path = self.root / str(record["crop"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read vehicle crop: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (160, 160):
            image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        image_tensor = (image_tensor - _MEAN) / _STD
        scalars = torch.tensor(
            [record["sph_score"], record["width_norm"], record["height_norm"]],
            dtype=torch.float32,
        )
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return image_tensor, scalars, label


def weighted_sampler_weights(labels: list[int]) -> list[float]:
    if not labels or any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be a non-empty binary list")
    counts = {label: labels.count(label) for label in (0, 1)}
    if not all(counts.values()):
        raise ValueError("both classes are required for weighted sampling")
    return [1.0 / counts[label] for label in labels]


def binary_average_precision(probabilities: list[float], labels: list[int]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    if any(not math.isfinite(value) for value in probabilities) or any(
        label not in {0, 1} for label in labels
    ):
        raise ValueError("probabilities must be finite and labels binary")
    positives = sum(labels)
    if positives == 0:
        raise ValueError("average precision requires at least one positive")
    ranked = sorted(range(len(labels)), key=lambda index: (-probabilities[index], index))
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(ranked, start=1):
        if labels[index] == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()
    probabilities: list[float] = []
    labels: list[int] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, scalars, targets in loader:
            logits = model(images.to(device), scalars.to(device))
            targets = targets.to(device)
            losses.append(float(criterion(logits, targets).item()))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(int(value) for value in targets.cpu().tolist())
    return binary_average_precision(probabilities, labels), sum(losses) / len(losses)


def train_vehicle_confirmer(
    dataset_root: Path,
    output_dir: Path,
    config: VehicleConfirmerTrainingConfig,
    device: str,
) -> Path:
    if not isinstance(config, VehicleConfirmerTrainingConfig):
        raise TypeError("config must be a VehicleConfirmerTrainingConfig")
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    report_path = dataset_root / "reports" / "vehicle-confirmer-dataset.json"
    if not report_path.is_file():
        raise ValueError(f"dataset report does not exist: {report_path}")
    _seed_everything(config.seed)
    train_dataset = _VehicleDataset(dataset_root, dataset_root / "manifests" / "train.jsonl")
    holdout_dataset = _VehicleDataset(dataset_root, dataset_root / "manifests" / "holdout.jsonl")
    labels = [int(record["label"]) for record in train_dataset.records]
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        weighted_sampler_weights(labels),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
    )
    torch_device = torch.device(device)
    model = VehicleConfirmer(pretrained=config.pretrained).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_ap = -1.0
    best_bce = math.inf
    metrics: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, scalars, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(torch_device), scalars.to(torch_device))
            loss = criterion(logits, targets.to(torch_device))
            if not torch.isfinite(loss):
                raise RuntimeError("vehicle confirmer loss is not finite")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        holdout_ap, holdout_bce = _evaluate(model, holdout_loader, torch_device)
        metrics.append(
            {
                "epoch": epoch,
                "train_bce": sum(train_losses) / len(train_losses),
                "holdout_ap": holdout_ap,
                "holdout_bce": holdout_bce,
            }
        )
        if holdout_ap > best_ap or (holdout_ap == best_ap and holdout_bce < best_bce):
            best_ap = holdout_ap
            best_bce = holdout_bce
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "holdout_ap": holdout_ap,
                    "holdout_bce": holdout_bce,
                    "dataset_report_sha256": _sha256(report_path),
                    "code_commit": _git_commit(Path(__file__).resolve().parents[3]),
                },
                best_path,
            )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "best.pt.sha256").write_text(_sha256(best_path) + "\n", encoding="utf-8")
    return best_path


def score_vehicle_confirmer(
    dataset_root: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str,
) -> tuple[dict[str, object], ...]:
    dataset = _VehicleDataset(Path(dataset_root), Path(manifest_path))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VehicleConfirmer(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    probabilities: list[float] = []
    with torch.no_grad():
        for images, scalars, _ in loader:
            probabilities.extend(
                torch.sigmoid(model(images.to(torch_device), scalars.to(torch_device)))
                .cpu()
                .tolist()
            )
    scored = tuple(
        {**record, "confirmation_probability": probability}
        for record, probability in zip(dataset.records, probabilities, strict=True)
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in scored),
        encoding="utf-8",
    )
    return scored


def export_vehicle_confirmer_onnx(
    checkpoint_path: Path, output_path: Path, device: str = "cpu"
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VehicleConfirmer(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            (torch.zeros(1, 3, 160, 160, device=device), torch.zeros(1, 3, device=device)),
            output_path,
            input_names=["images", "scalar_features"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "scalar_features": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX export requires the onnx package") from exc
    return output_path


def export_vehicle_confirmer_engine(onnx_path: Path, engine_path: Path) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--minShapes=images:1x3x160x160,scalar_features:1x3",
        "--optShapes=images:128x3x160x160,scalar_features:128x3",
        "--maxShapes=images:512x3x160x160,scalar_features:512x3",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("TensorRT export requires trtexec on PATH") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec failed: {completed.stderr.strip()}")
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError("trtexec did not create a non-empty engine")
    metadata = {"command": command, "engine_sha256": _sha256(engine_path)}
    engine_path.with_suffix(engine_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return engine_path
