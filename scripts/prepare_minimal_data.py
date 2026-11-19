#!/usr/bin/env python3
"""Prepare TreeWeave Minimal Experiment Datasets (`ms-swift` JSONL with verified absolute media paths).

Two modes:
1. `--mode smoke`:
   Zero-download synthetic smoke dataset (high-whitespace document/GUI/natural PNGs + multi-frame MP4 videos
   containing moving objects, static dense text regions, and static background) for 5-minute pipeline validation
   on a fresh dev/GPU machine.
2. `--mode convert`:
   Converts locally downloaded HuggingFace / ModelScope datasets (Parquet, Arrow, JSON, JSONL) into a single
   balanced, deduplicated `ms-swift` prompt JSONL ready for `scripts/02_offline_rollout_8xh100.sh`.

Usage examples:
  # 1) Zero-download 5-minute smoke set (64 images + 32 videos):
  python scripts/prepare_minimal_data.py --mode smoke --output-dir ./data/smoke --num-images 64 --num-videos 32

  # 2) Convert locally downloaded real datasets into a 24K minimal validation mix:
  python scripts/prepare_minimal_data.py --mode convert \
    --output-jsonl ./data/minimal_train_prompts.jsonl \
    --media-out-dir ./data/extracted_media \
    --doc-dirs /data/DocVQA /data/ChartQA \
    --general-dirs /data/LLaVA-OneVision-subset \
    --gui-dirs /data/UGround-subset \
    --video-dirs /data/LLaVA-Video-178K-subset \
    --doc-limit 8000 --general-limit 4000 --gui-limit 4000 --video-limit 8000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _write_mp4_pyav(frames_rgb: Sequence[np.ndarray], out_path: Path, fps: int = 2) -> None:
    """Write standard H.264/MPEG-4 `.mp4` video using PyAV (zero external ffmpeg binary needed)."""
    import av

    h, w = int(frames_rgb[0].shape[0]), int(frames_rgb[0].shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), mode="w") as container:
        try:
            stream = container.add_stream("libx264", rate=fps)
        except Exception:
            stream = container.add_stream("mpeg4", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for arr in frames_rgb:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _generate_smoke_dataset(
    output_dir: Path,
    *,
    num_images: int = 64,
    num_videos: int = 32,
    video_frames: int = 8,
    seed: int = 42,
) -> Path:
    """Create a realistic high-contrast smoke set (docs with whitespace + GUI + static/moving videos)."""
    rng = random.Random(seed)
    output_dir = output_dir.resolve()
    img_dir = output_dir / "images"
    vid_dir = output_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    img_prompts = [
        "<image> Read the document header and report the highlighted value concisely.",
        "<image> Locate the primary button on this desktop screen and describe its position.",
        "<image> Summarize the key visual elements in this image in one or two sentences.",
    ]
    for i in range(num_images):
        kind = i % 3
        w, h = (512, 512) if kind != 1 else (640, 384)
        bg = (250, 250, 252) if kind == 0 else ((238, 242, 248) if kind == 1 else (40 + i % 40, 80, 140))
        im = Image.new("RGB", (w, h), color=bg)
        draw = ImageDraw.Draw(im)
        # Concentrated dense region + large uniform whitespace/background to exercise s=0/1/2 quadtree
        if kind == 0:
            draw.rectangle([28, 24, 240, 140], outline=(30, 58, 138), width=2)
            draw.text((40, 38), f"INVOICE #{1000 + i}", fill=(15, 23, 42))
            draw.text((40, 64), f"TOTAL: ${120 + i * 7}.50", fill=(220, 38, 38))
            draw.text((40, 92), "STATUS: APPROVED", fill=(13, 148, 136))
        elif kind == 1:
            bx = rng.randint(40, w - 160)
            by = rng.randint(40, h - 100)
            draw.rectangle([bx, by, bx + 110, by + 36], fill=(37, 99, 235), outline=(29, 78, 216), width=2)
            draw.text((bx + 24, by + 12), f"SUBMIT {i}", fill=(255, 255, 255))
        else:
            cx, cy = rng.randint(96, w - 96), rng.randint(96, h - 96)
            draw.ellipse([cx - 48, cy - 48, cx + 48, cy + 48], fill=(234, 88, 12), outline=(255, 255, 255), width=3)
            draw.rectangle([20, h - 70, 210, h - 24], fill=(15, 23, 42))
            draw.text((32, h - 54), f"SCENE TAG {i}", fill=(255, 255, 255))

        img_path = img_dir / f"smoke_img_{i:04d}.png"
        im.save(img_path)
        rows.append(
            {
                "messages": [{"role": "user", "content": img_prompts[kind]}],
                "images": [str(img_path)],
                "source_domain": ["doc", "gui", "general"][kind],
            }
        )

    vid_prompts = [
        "<video> Read the static caption on the slide and describe how the colored square moves across frames.",
        "<video> What stays stationary in the video and what changes over time?",
    ]
    for i in range(num_videos):
        frames: List[np.ndarray] = []
        w, h = 256, 256
        for t in range(video_frames):
            im = Image.new("RGB", (w, h), color=(244, 246, 250))
            draw = ImageDraw.Draw(im)
            # Static dense text banner (tests orthogonal s=0, t=2 static fine text!)
            draw.rectangle([12, 12, 244, 62], fill=(255, 255, 255), outline=(30, 58, 138), width=2)
            draw.text((22, 22), f"SLIDE #{i:02d} STATIC TEXT", fill=(15, 23, 42))
            draw.text((22, 40), f"CODE: TW-{2026 + i}", fill=(37, 99, 235))
            # Moving foreground block (tests dynamic s=0, t=0 breaking temporal merge!)
            mx = 24 + int((w - 84) * (t / max(video_frames - 1, 1)))
            my = 110 + (18 if (t % 2 == 0) else -12)
            draw.rectangle([mx, my, mx + 36, my + 36], fill=(220, 38, 38), outline=(127, 29, 29), width=2)
            frames.append(np.asarray(im, dtype=np.uint8))

        vid_path = vid_dir / f"smoke_vid_{i:04d}.mp4"
        _write_mp4_pyav(frames, vid_path, fps=2)
        rows.append(
            {
                "messages": [{"role": "user", "content": vid_prompts[i % len(vid_prompts)]}],
                "videos": [str(vid_path)],
                "source_domain": "video",
            }
        )

    rng.shuffle(rows)
    out_jsonl = output_dir / "smoke_prompts.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[prepare_minimal_data] Wrote {len(rows)} smoke prompts "
        f"({num_images} images + {num_videos} videos) -> {out_jsonl}"
    )
    return out_jsonl


def _ensure_tag(text: str, tag: str, count: int) -> str:
    clean = (text or "").strip()
    existing = clean.count(tag)
    if existing >= count:
        return clean
    prefix = "".join(f"{tag}\n" for _ in range(count - existing))
    return f"{prefix}{clean}" if clean else prefix.strip()


def _materialize_image(obj: Any, base_dir: Path, media_out_dir: Path) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        p = Path(obj)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p) if p.is_file() else None
    if isinstance(obj, dict):
        if obj.get("path") and Path(obj["path"]).is_file():
            return str(Path(obj["path"]).resolve())
        if obj.get("bytes"):
            raw = obj["bytes"]
            digest = hashlib.sha1(raw).hexdigest()[:16]
            out = media_out_dir / f"img_{digest}.png"
            if not out.is_file():
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.open(io.BytesIO(raw)).convert("RGB").save(out)
            return str(out.resolve())
    if isinstance(obj, Image.Image):
        buf = io.BytesIO()
        obj.convert("RGB").save(buf, format="PNG")
        raw = buf.getvalue()
        digest = hashlib.sha1(raw).hexdigest()[:16]
        out = media_out_dir / f"img_{digest}.png"
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
        return str(out.resolve())
    return None


def _extract_prompt_text(rec: Dict[str, Any]) -> Optional[str]:
    if isinstance(rec.get("messages"), list):
        for m in rec["messages"]:
            if m.get("role") == "user" and m.get("content"):
                return str(m["content"])
    if isinstance(rec.get("conversations"), list):
        for turn in rec["conversations"]:
            role = str(turn.get("from", "")).lower()
            if role in ("human", "user") and turn.get("value"):
                return str(turn["value"])
    for key in ("question", "prompt", "query", "instruction", "text"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _iter_records_from_path(src: Path) -> Iterable[Tuple[Dict[str, Any], Path]]:
    if src.is_file():
        files = [src]
    else:
        files = sorted(
            p
            for p in src.rglob("*")
            if p.is_file() and p.suffix.lower() in (".jsonl", ".json", ".parquet", ".arrow")
        )
    for fp in files:
        suf = fp.suffix.lower()
        if suf == ".jsonl":
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line), fp.parent
        elif suf == ".json":
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item, fp.parent
            elif isinstance(data, dict) and isinstance(data.get("data"), list):
                for item in data["data"]:
                    if isinstance(item, dict):
                        yield item, fp.parent
        elif suf in (".parquet", ".arrow"):
            from datasets import Dataset

            ds = Dataset.from_parquet(str(fp)) if suf == ".parquet" else Dataset.from_file(str(fp))
            for item in ds:
                yield dict(item), fp.parent


def _collect_domain_rows(
    sources: Sequence[Path],
    *,
    domain: str,
    limit: int,
    media_out_dir: Path,
    seed: int,
) -> List[Dict[str, Any]]:
    if limit <= 0 or not sources:
        return []
    collected: List[Dict[str, Any]] = []
    seen_keys = set()

    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {src}")
        for rec, base_dir in _iter_records_from_path(src):
            prompt = _extract_prompt_text(rec)
            if not prompt:
                continue

            # Resolve videos first if present
            raw_vids = rec.get("videos") if isinstance(rec.get("videos"), list) else (
                [rec["video"]] if rec.get("video") else []
            )
            vids: List[str] = []
            for v in raw_vids:
                if isinstance(v, str):
                    vp = Path(v)
                    if not vp.is_absolute():
                        vp = (base_dir / vp).resolve()
                    if vp.is_file():
                        vids.append(str(vp))

            raw_imgs = rec.get("images") if isinstance(rec.get("images"), list) else (
                [rec["image"]] if rec.get("image") is not None else []
            )
            imgs: List[str] = []
            for im_obj in raw_imgs:
                resolved = _materialize_image(im_obj, base_dir, media_out_dir)
                if resolved:
                    imgs.append(resolved)

            if not vids and not imgs:
                continue

            if vids:
                prompt_tagged = _ensure_tag(prompt, "<video>", len(vids))
                out_row: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": prompt_tagged}],
                    "videos": vids,
                    "source_domain": domain,
                }
            else:
                prompt_tagged = _ensure_tag(prompt, "<image>", len(imgs))
                out_row = {
                    "messages": [{"role": "user", "content": prompt_tagged}],
                    "images": imgs,
                    "source_domain": domain,
                }

            dedup_key = hashlib.sha1(
                json.dumps(out_row, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            collected.append(out_row)

    rng = random.Random(seed)
    rng.shuffle(collected)
    return collected[:limit]


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "convert"], required=True)
    p.add_argument("--output-dir", type=Path, default=Path("./data/smoke"))
    p.add_argument("--output-jsonl", type=Path, default=Path("./data/minimal_train_prompts.jsonl"))
    p.add_argument("--media-out-dir", type=Path, default=Path("./data/extracted_media"))
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--num-videos", type=int, default=32)
    p.add_argument("--video-frames", type=int, default=8)
    p.add_argument("--doc-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--general-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--gui-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--video-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--doc-limit", type=int, default=8000)
    p.add_argument("--general-limit", type=int, default=4000)
    p.add_argument("--gui-limit", type=int, default=4000)
    p.add_argument("--video-limit", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.mode == "smoke":
        _generate_smoke_dataset(
            args.output_dir,
            num_images=args.num_images,
            num_videos=args.num_videos,
            video_frames=args.video_frames,
            seed=args.seed,
        )
        return

    media_out = args.media_out_dir.resolve()
    media_out.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for domain, dirs, lim in (
        ("doc", args.doc_dirs, args.doc_limit),
        ("general", args.general_dirs, args.general_limit),
        ("gui", args.gui_dirs, args.gui_limit),
        ("video", args.video_dirs, args.video_limit),
    ):
        part = _collect_domain_rows(dirs, domain=domain, limit=lim, media_out_dir=media_out, seed=args.seed)
        counts[domain] = len(part)
        all_rows.extend(part)

    if not all_rows:
        raise RuntimeError("No valid multimodal samples were extracted. Check your `--*-dirs` paths.")

    random.Random(args.seed).shuffle(all_rows)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[prepare_minimal_data] Wrote {len(all_rows)} rows {counts} -> {args.output_jsonl.resolve()}")


if __name__ == "__main__":
    main()
