from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from imagecorruptions import corrupt
from imagecorruptions import corruptions as corruption_impl
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage import color
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_C_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGE_PREPROCESSORS = (
    "identity",
    "gaussian_0.5",
    "gaussian_1.0",
    "gaussian_1.5",
    "gaussian_2.0",
    "gaussian_2.5",
    "median_3",
    "unsharp_1",
    "unsharp_2",
    "unsharp_r1_p100",
    "unsharp_r1_p300",
    "unsharp_r2_p300",
    "unsharp_r3_p300",
    "sharpness_2",
    "sharpness_3",
    "highboost_05",
    "highboost_10",
    "autocontrast",
    "autocontrast_1",
    "autocontrast_2",
    "contrast_125",
    "contrast_150",
    "contrast_200",
    "brightness_070",
    "brightness_080",
    "brightness_090",
    "gamma_120",
    "gamma_150",
    "equalize",
    "dehaze_07",
    "dehaze_09",
    "inverse_defocus_05",
    "inverse_defocus_10",
    "inverse_defocus_15",
    "inverse_defocus_20",
    "inverse_motion_025",
    "inverse_motion_050",
    "inverse_motion_100",
    "inverse_zoom_05",
    "inverse_zoom_10",
    "inverse_brightness_025",
    "inverse_brightness_050",
    "inverse_brightness_c055",
    "inverse_brightness_c065",
    "inverse_brightness_c075",
    "inverse_brightness_c085",
    "inverse_brightness_c055_gamma090",
    "inverse_brightness_c055_gamma095",
    "inverse_brightness_c055_autocontrast1",
    "snow_suppress_05",
    "snow_suppress_10",
    "frost_ac2_gaussian05",
    "frost_ac2_median3",
    "fog_contrast_175",
    "fog_contrast_200",
)


def preprocess_image(image: Image.Image, name: str) -> Image.Image:
    """Apply a deterministic, lightweight image-space preprocessing operator."""
    if name == "identity":
        return image
    if name.startswith("inverse_brightness_c055_"):
        restored = preprocess_image(image, "inverse_brightness_c055")
        suffix = name.removeprefix("inverse_brightness_c055_")
        if suffix == "gamma090":
            lut = [round(255 * (value / 255) ** 0.90) for value in range(256)]
            return restored.point(lut * 3)
        if suffix == "gamma095":
            lut = [round(255 * (value / 255) ** 0.95) for value in range(256)]
            return restored.point(lut * 3)
        if suffix == "autocontrast1":
            return ImageOps.autocontrast(restored, cutoff=1)
        raise ValueError(f"unsupported censored-brightness postprocessor: {name}")
    if name.startswith("gaussian_"):
        return image.filter(ImageFilter.GaussianBlur(radius=float(name.split("_")[1])))
    if name == "median_3":
        return image.filter(ImageFilter.MedianFilter(size=3))
    if name == "unsharp_1":
        return image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    if name == "unsharp_2":
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    unsharp = {
        "unsharp_r1_p100": (1, 100),
        "unsharp_r1_p300": (1, 300),
        "unsharp_r2_p300": (2, 300),
        "unsharp_r3_p300": (3, 300),
    }
    if name in unsharp:
        radius, percent = unsharp[name]
        return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=0))
    if name.startswith("sharpness_"):
        return ImageEnhance.Sharpness(image).enhance(float(name.split("_")[1]))
    if name.startswith("highboost_"):
        amount = {"05": 0.5, "10": 1.0}[name.rsplit("_", 1)[1]]
        original = np.asarray(image, dtype=np.float32)
        lowpass = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
        restored = np.clip(original + amount * (original - lowpass), 0, 255).astype(np.uint8)
        return Image.fromarray(restored, mode="RGB")
    if name == "autocontrast":
        return ImageOps.autocontrast(image)
    if name.startswith("autocontrast_"):
        return ImageOps.autocontrast(image, cutoff=int(name.rsplit("_", 1)[1]))
    if name.startswith("contrast_"):
        factor = {"125": 1.25, "150": 1.50, "200": 2.00}[name.rsplit("_", 1)[1]]
        return ImageEnhance.Contrast(image).enhance(factor)
    if name.startswith("brightness_"):
        factor = {"070": 0.70, "080": 0.80, "090": 0.90}[name.rsplit("_", 1)[1]]
        return ImageEnhance.Brightness(image).enhance(factor)
    if name.startswith("gamma_"):
        gamma = {"120": 1.20, "150": 1.50}[name.rsplit("_", 1)[1]]
        lut = [round(255 * (value / 255) ** gamma) for value in range(256)]
        return image.point(lut * 3)
    if name == "equalize":
        return ImageOps.equalize(image)
    if name.startswith("dehaze_"):
        omega = {"07": 0.7, "09": 0.9}[name.rsplit("_", 1)[1]]
        array = np.asarray(image, dtype=np.float32) / 255.0
        dark = Image.fromarray((array.min(axis=2) * 255).astype(np.uint8), mode="L")
        dark = np.asarray(dark.filter(ImageFilter.MinFilter(size=15)), dtype=np.float32) / 255.0
        count = max(1, dark.size // 1000)
        brightest = np.argpartition(dark.reshape(-1), -count)[-count:]
        atmosphere = array.reshape(-1, 3)[brightest].mean(axis=0).clip(0.1, 1.0)
        transmission = 1.0 - omega * dark / float(atmosphere.max())
        transmission = np.maximum(transmission, 0.20)[..., None]
        restored = (array - atmosphere) / transmission + atmosphere
        return Image.fromarray(np.uint8(np.clip(restored, 0.0, 1.0) * 255), mode="RGB")
    if name.startswith("inverse_defocus_"):
        amount = {"05": 0.5, "10": 1.0, "15": 1.5, "20": 2.0}[
            name.rsplit("_", 1)[1]
        ]
        array = np.asarray(image, dtype=np.float32)
        reblurred = corruption_impl.defocus_blur(array, severity=5).astype(np.float32)
        restored = np.clip(array + amount * (array - reblurred), 0, 255)
        return Image.fromarray(restored.astype(np.uint8), mode="RGB")
    if name.startswith("inverse_motion_"):
        amount = {"025": 0.25, "050": 0.5, "100": 1.0}[name.rsplit("_", 1)[1]]
        array = np.asarray(image, dtype=np.float32)
        candidates = []
        for angle in (-45, -30, -15, 0, 15, 30, 45):
            reblurred = corruption_impl._motion_blur(
                array, radius=20, sigma=15, angle=angle
            ).astype(np.float32)
            score = float(np.mean(np.abs(array - reblurred)))
            candidates.append((score, reblurred))
        reblurred = min(candidates, key=lambda item: item[0])[1]
        restored = np.clip(array + amount * (array - reblurred), 0, 255)
        return Image.fromarray(restored.astype(np.uint8), mode="RGB")
    if name.startswith("inverse_zoom_"):
        amount = {"05": 0.5, "10": 1.0}[name.rsplit("_", 1)[1]]
        array = np.asarray(image, dtype=np.float32)
        reblurred = corruption_impl.zoom_blur(array, severity=5).astype(np.float32)
        restored = np.clip(array + amount * (array - reblurred), 0, 255)
        return Image.fromarray(restored.astype(np.uint8), mode="RGB")
    if name.startswith("inverse_brightness_"):
        suffix = name.rsplit("_", 1)[1]
        array = np.asarray(image, dtype=np.float32) / 255.0
        hsv = color.rgb2hsv(array)
        if suffix.startswith("c"):
            # Severity-5 brightness adds 0.5 to HSV value and clips at one.
            # The unclipped region is exactly invertible; the censored region
            # uses a development-selected neutral value prior rather than
            # incorrectly applying the same subtraction everywhere.
            clipped_value = {"c055": 0.55, "c065": 0.65, "c075": 0.75, "c085": 0.85}[suffix]
            observed = hsv[..., 2]
            hsv[..., 2] = np.where(
                observed >= (1.0 - 0.5 / 255.0),
                clipped_value,
                np.clip(observed - 0.50, 0.0, 1.0),
            )
        else:
            subtraction = {"025": 0.25, "050": 0.50}[suffix]
            hsv[..., 2] = np.clip(hsv[..., 2] - subtraction, 0.0, 1.0)
        restored = color.hsv2rgb(hsv)
        return Image.fromarray(np.uint8(np.clip(restored, 0.0, 1.0) * 255), mode="RGB")
    if name.startswith("snow_suppress_"):
        amount = {"05": 0.5, "10": 1.0}[name.rsplit("_", 1)[1]]
        array = np.asarray(image, dtype=np.float32)
        median = np.asarray(image.filter(ImageFilter.MedianFilter(size=3)), dtype=np.float32)
        value = array.max(axis=2)
        local_value = median.max(axis=2)
        mask = np.clip((value - local_value - 12.0) / 40.0, 0.0, 1.0)[..., None]
        restored = array * (1.0 - amount * mask) + median * (amount * mask)
        # Severity-5 snow also raises the bright background; a mild inverse tone step is stable.
        restored = np.uint8(np.clip(restored, 0, 255))
        lut = [round(255 * (value / 255) ** 1.2) for value in range(256)]
        return Image.fromarray(restored, mode="RGB").point(lut * 3)
    if name == "frost_ac2_gaussian05":
        return ImageOps.autocontrast(
            image.filter(ImageFilter.GaussianBlur(radius=0.5)), cutoff=2
        )
    if name == "frost_ac2_median3":
        return ImageOps.autocontrast(image.filter(ImageFilter.MedianFilter(size=3)), cutoff=2)
    if name.startswith("fog_contrast_"):
        factor = {"175": 1.75, "200": 2.0}[name.rsplit("_", 1)[1]]
        return ImageEnhance.Contrast(image).enhance(factor)
    raise ValueError(f"unsupported preprocessor: {name}")


def estimate_noise_level(normalized_images: torch.Tensor) -> torch.Tensor:
    """Immerkaer blind additive-noise estimate for CLIP-normalized RGB tensors."""
    if normalized_images.ndim == 3:
        normalized_images = normalized_images[None]
    if normalized_images.ndim != 4 or normalized_images.shape[1] != 3:
        raise ValueError("expected [batch, 3, height, width] images")
    mean = normalized_images.new_tensor(CLIP_MEAN)[None, :, None, None]
    std = normalized_images.new_tensor(CLIP_STD)[None, :, None, None]
    images = (normalized_images * std + mean).clamp(0.0, 1.0)
    kernel = normalized_images.new_tensor(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]]
    )[None, None].repeat(3, 1, 1, 1)
    residual = F.conv2d(images, kernel, groups=3)
    height, width = images.shape[-2:]
    normalizer = 6.0 * 3.0 * (height - 2) * (width - 2)
    return math.sqrt(math.pi / 2.0) * residual.abs().sum(dim=(1, 2, 3)) / normalizer


# Preserve the original ImageNet-C formulas with current NumPy/scikit-image.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
_skimage_gaussian = corruption_impl.gaussian


def _gaussian_compat(*args, multichannel=None, **kwargs):
    if multichannel is not None and "channel_axis" not in kwargs:
        kwargs["channel_axis"] = -1 if multichannel else None
    return _skimage_gaussian(*args, **kwargs)


corruption_impl.gaussian = _gaussian_compat


def _stable_seed(seed: int, relative_path: str, corruption: str) -> int:
    payload = f"{seed}|{relative_path}|{corruption}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def _deterministic_impulse_noise(
    image: np.ndarray, severity: int, seed: int
) -> np.ndarray:
    """Apply ImageNet-C salt-and-pepper noise with an explicit local RNG.

    Current scikit-image releases create an unseeded ``default_rng`` inside
    ``random_noise``.  Seeding NumPy's legacy global state therefore does not
    control ImageNet-C's original ``impulse_noise`` wrapper.  This implements
    the same formula while making the per-image seed explicit and isolated.
    """
    amount = (0.03, 0.06, 0.09, 0.17, 0.27)[severity - 1]
    rng = np.random.default_rng(seed)
    normalized = image.astype(np.float64) / 255.0
    flipped = rng.random(normalized.shape) <= amount
    salted = rng.random(normalized.shape) <= 0.5
    output = normalized.copy()
    output[flipped & salted] = 1.0
    output[flipped & ~salted] = 0.0
    return np.uint8(np.clip(output, 0.0, 1.0) * 255.0)


def stratified_subset(
    relative_paths: Sequence[str], n: int | None, seed: int, offset: int = 0
) -> list[str]:
    """Select a deterministic approximately class-balanced subset or fold.

    ``offset`` slices a later, disjoint segment from the same stratified order.
    It is useful for confirmation folds that must not share source images.
    """
    paths = [p.strip() for p in relative_paths if p.strip()]
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if n is None:
        if offset:
            raise ValueError("offset requires a finite subset size")
        return sorted(paths)
    if n <= 0:
        raise ValueError("n must be positive")
    if offset + n > len(paths):
        raise ValueError("offset + n exceeds the available paths")
    if n >= len(paths) and offset == 0:
        return sorted(paths)

    by_class: dict[str, list[str]] = {}
    for path in paths:
        by_class.setdefault(path.split("/", 1)[0], []).append(path)
    rng = np.random.default_rng(seed)
    for values in by_class.values():
        rng.shuffle(values)

    classes = sorted(by_class)
    rng.shuffle(classes)
    selected: list[str] = []
    cursor = 0
    while len(selected) < offset + n:
        cls = classes[cursor % len(classes)]
        row = cursor // len(classes)
        if row < len(by_class[cls]):
            selected.append(by_class[cls][row])
        cursor += 1
    return selected[offset : offset + n]


def stratified_disjoint_folds(
    relative_paths: Sequence[str], fold_size: int, fold_count: int, seed: int
) -> list[list[str]]:
    """Build deterministic, mutually disjoint, approximately balanced folds.

    Samples from each class are first rotated across disjoint pools.  Each pool
    is then stratified independently, avoiding the progressively poorer class
    coverage produced by consecutive slices of one global ordering.
    """
    paths = [path.strip() for path in relative_paths if path.strip()]
    if fold_size <= 0 or fold_count <= 1:
        raise ValueError("fold_size must be positive and fold_count must exceed one")
    if fold_size * fold_count > len(paths):
        raise ValueError("requested folds exceed the available paths")

    by_class: dict[str, list[str]] = {}
    for path in paths:
        by_class.setdefault(path.split("/", 1)[0], []).append(path)
    rng = np.random.default_rng(seed)
    classes = sorted(by_class)
    rng.shuffle(classes)
    pools: list[list[str]] = [[] for _ in range(fold_count)]
    for class_index, cls in enumerate(classes):
        values = by_class[cls]
        rng.shuffle(values)
        start = class_index % fold_count
        for value_index, path in enumerate(values):
            pools[(start + value_index) % fold_count].append(path)

    if any(len(pool) < fold_size for pool in pools):
        raise ValueError("a stratified pool is smaller than the requested fold")
    return [
        stratified_subset(pool, fold_size, seed + 1009 * (index + 1))
        for index, pool in enumerate(pools)
    ]


class OnlineImageNetCorruption(Dataset):
    """Generate ImageNet-C-compatible 224x224 corruptions without storing copies."""

    def __init__(
        self,
        root: str | Path,
        relative_paths: Sequence[str],
        class_to_idx: dict[str, int],
        corruption: str,
        severity: int = 5,
        seed: int = 17,
        preprocessor: str = "identity",
        cache_dir: str | Path | None = None,
    ) -> None:
        if corruption != "clean" and corruption not in IMAGENET_C_CORRUPTIONS:
            raise ValueError(f"unsupported corruption: {corruption}")
        if not 1 <= severity <= 5:
            raise ValueError("severity must be in [1, 5]")
        if preprocessor not in IMAGE_PREPROCESSORS:
            raise ValueError(f"unsupported preprocessor: {preprocessor}")
        self.root = Path(root)
        self.relative_paths = list(relative_paths)
        self.class_to_idx = class_to_idx
        self.corruption = corruption
        self.severity = severity
        self.seed = seed
        self.preprocessor = preprocessor
        self._cached_images = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            protocol_path = cache_dir / "protocol.json"
            protocol = json.loads(protocol_path.read_text())
            ids_sha256 = hashlib.sha256("\n".join(self.relative_paths).encode()).hexdigest()
            key = f"{self.corruption}_s{self.severity}"
            checks = {
                "ids": protocol.get("ids_sha256") == ids_sha256,
                "n": protocol.get("n") == len(self.relative_paths),
                "seed": protocol.get("corruption_seed") == self.seed,
                "scenario": key in protocol.get("files", {}),
            }
            if not all(checks.values()):
                raise RuntimeError(f"corruption cache mismatch: {checks}")
            cache_path = cache_dir / protocol["files"][key]["path"]
            self._cached_images = np.load(cache_path, mmap_mode="r")
            if self._cached_images.shape != (len(self.relative_paths), 224, 224, 3):
                raise RuntimeError("cached image array has an invalid shape")

    def __len__(self) -> int:
        return len(self.relative_paths)

    def _load_corrupted_image(self, index: int) -> tuple[Image.Image, int, int]:
        relative_path = self.relative_paths[index]
        if self._cached_images is not None:
            wnid = relative_path.split("/", 1)[0]
            image = Image.fromarray(np.asarray(self._cached_images[index]), mode="RGB")
            return image, self.class_to_idx[wnid], index
        path = self.root / relative_path
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        # ImageNet-C's public generator used torchvision's default PIL bilinear
        # resize before applying the corruption. CLIP's bicubic resize happens
        # later but is an identity for these already-224px images.
        image = TF.resize(image, 256, interpolation=InterpolationMode.BILINEAR)
        image = TF.center_crop(image, [224, 224])

        if self.corruption != "clean":
            array = np.asarray(image, dtype=np.uint8)
            local_seed = _stable_seed(self.seed, relative_path, self.corruption)
            if self.corruption == "impulse_noise":
                array = _deterministic_impulse_noise(
                    array, severity=self.severity, seed=local_seed
                )
            else:
                state = np.random.get_state()
                np.random.seed(local_seed)
                try:
                    array = corrupt(
                        array,
                        corruption_name=self.corruption,
                        severity=self.severity,
                    )
                finally:
                    np.random.set_state(state)
            image = Image.fromarray(array.astype(np.uint8), mode="RGB")

        wnid = relative_path.split("/", 1)[0]
        return image, self.class_to_idx[wnid], index

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, CLIP_MEAN, CLIP_STD)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label, sample_id = self._load_corrupted_image(index)
        image = preprocess_image(image, self.preprocessor)
        return self._to_tensor(image), label, sample_id


class OnlineImageNetMultiViewCorruption(OnlineImageNetCorruption):
    """Generate one corruption, then derive an optionally gated filter bank."""

    def __init__(self, *args, preprocessors: Sequence[str], noise_gate_threshold: float, **kwargs):
        super().__init__(*args, preprocessor="identity", **kwargs)
        if not preprocessors:
            raise ValueError("at least one preprocessor is required")
        invalid = [value for value in preprocessors if value not in IMAGE_PREPROCESSORS]
        if invalid:
            raise ValueError(f"unsupported preprocessors: {invalid}")
        if preprocessors[0] != "identity":
            raise ValueError("the first multiview preprocessor must be identity")
        self.preprocessors = tuple(preprocessors)
        self.noise_gate_threshold = float(noise_gate_threshold)

    def __getitem__(self, index: int):
        image, label, sample_id = self._load_corrupted_image(index)
        identity = self._to_tensor(image)
        routed = (
            self.noise_gate_threshold < 0
            or estimate_noise_level(identity)[0].item() >= self.noise_gate_threshold
        )
        if routed:
            views = [self._to_tensor(preprocess_image(image, name)) for name in self.preprocessors]
        else:
            views = [identity for _ in self.preprocessors]
        return torch.stack(views), label, sample_id


class TaggedConcatDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = list(datasets)
        self.index: list[tuple[int, int]] = []
        for domain, dataset in enumerate(self.datasets):
            self.index.extend((domain, sample) for sample in range(len(dataset)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        domain, sample = self.index[index]
        image, label, sample_id = self.datasets[domain][sample]
        return image, label, domain, sample_id


class MultiViewDataset(Dataset):
    """Align deterministic preprocessing views without storing image copies."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        if not datasets:
            raise ValueError("at least one view is required")
        if len({len(dataset) for dataset in datasets}) != 1:
            raise ValueError("all views must contain the same samples")
        self.datasets = list(datasets)

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, index: int):
        rows = [dataset[index] for dataset in self.datasets]
        labels = {row[1] for row in rows}
        sample_ids = {row[2] for row in rows}
        if len(labels) != 1 or len(sample_ids) != 1:
            raise RuntimeError("preprocessing views are not aligned")
        return torch.stack([row[0] for row in rows]), rows[0][1], rows[0][2]
