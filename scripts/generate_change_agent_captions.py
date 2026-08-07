"""Generate one Change-Agent caption per LEVIR-MCI image pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List

MEAN = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
STD = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_openmmlab_import_stubs() -> None:
    """The upstream SegFormer imports two helpers that inference never calls."""

    mmseg = types.ModuleType("mmseg")
    mmseg_utils = types.ModuleType("mmseg.utils")
    mmseg_utils.get_root_logger = lambda: logging.getLogger("change-agent")
    mmseg.utils = mmseg_utils

    mmcv = types.ModuleType("mmcv")
    mmcv_runner = types.ModuleType("mmcv.runner")

    def unsupported_load_checkpoint(*args: object, **kwargs: object) -> None:
        raise RuntimeError("mmcv.load_checkpoint is not used by the MCI inference path")

    mmcv_runner.load_checkpoint = unsupported_load_checkpoint
    mmcv.runner = mmcv_runner
    sys.modules.setdefault("mmseg", mmseg)
    sys.modules.setdefault("mmseg.utils", mmseg_utils)
    sys.modules.setdefault("mmcv", mmcv)
    sys.modules.setdefault("mmcv.runner", mmcv_runner)


def read_names(list_path: Path, limit: int) -> List[str]:
    names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if limit < 1:
        raise ValueError("limit must be at least one")
    if limit > len(names):
        raise ValueError("limit {} exceeds split size {}".format(limit, len(names)))
    return names[:limit]


def decode_caption(sequence: Iterable[int], vocabulary: Dict[str, int]) -> str:
    ignored = {vocabulary["<START>"], vocabulary["<END>"], vocabulary["<NULL>"]}
    index_to_word = {index: word for word, index in vocabulary.items()}
    words = [index_to_word[index] for index in sequence if index not in ignored]
    return " ".join(words).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_root = args.source_root.resolve()
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    list_dir = source_root / "data" / "LEVIR_MCI"
    names = read_names(list_dir / "{}.txt".format(args.split), args.limit)
    vocabulary = json.loads((list_dir / "vocab.json").read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict):
        raise ValueError("vocab.json must contain an object")

    sys.path.insert(0, str(source_root))
    install_openmmlab_import_stubs()

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from model.model_decoder import DecoderTransformer
    from model.model_encoder_att import AttentiveEncoder, Encoder

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the upstream decoder requires a CUDA device")

    started_at = time.time()
    checkpoint_hash = sha256_file(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = Encoder("segformer-mit_b1")
    encoder_trans = AttentiveEncoder(
        train_stage=None,
        n_layers=3,
        feature_size=[16, 16, 512],
        heads=8,
        dropout=0.1,
    )
    decoder = DecoderTransformer(
        encoder_dim=512,
        feature_dim=512,
        vocab_size=len(vocabulary),
        max_lengths=41,
        word_vocab=vocabulary,
        n_head=8,
        n_layers=1,
        dropout=0.1,
    )
    encoder.load_state_dict(state["encoder_dict"])
    encoder_trans.load_state_dict(state["encoder_trans_dict"], strict=False)
    decoder.load_state_dict(state["decoder_dict"])
    models = (
        encoder.to(args.device).eval(),
        encoder_trans.to(args.device).eval(),
        decoder.to(args.device).eval(),
    )

    def preprocess(path: Path):
        image = np.asarray(imageio.imread(path), dtype=np.float32)
        if image.shape != (256, 256, 3):
            raise ValueError("unexpected image shape {} for {}".format(image.shape, path))
        image = image.transpose(2, 0, 1)
        for channel in range(3):
            image[channel] = (image[channel] - MEAN[channel]) / STD[channel]
        return torch.from_numpy(image).unsqueeze(0).to(args.device)

    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    empty_captions = 0
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        with torch.inference_mode():
            for index, filename in enumerate(names, start=1):
                image_a = dataset_root / "images" / args.split / "A" / filename
                image_b = dataset_root / "images" / args.split / "B" / filename
                tensor_a = preprocess(image_a)
                tensor_b = preprocess(image_b)
                features_a, features_b = models[0](tensor_a, tensor_b)
                features_a, features_b, _ = models[1](features_a, features_b)
                sequence = models[2].sample(features_a, features_b, k=1)
                caption = decode_caption(sequence, vocabulary)
                empty_captions += int(not caption)
                record = {
                    "item_id": Path(filename).stem,
                    "original_caption": caption,
                    "source": "Change-Agent/MCI_model.pth",
                    "split": args.split,
                    "image_a": str(image_a),
                    "image_b": str(image_b),
                    "checkpoint_sha256": checkpoint_hash,
                    "sample_index": index,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records += 1
                print("[{}/{}] {}: {}".format(index, len(names), record["item_id"], caption))

    manifest = {
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "split": args.split,
        "requested_limit": args.limit,
        "record_count": records,
        "empty_caption_count": empty_captions,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if empty_captions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
