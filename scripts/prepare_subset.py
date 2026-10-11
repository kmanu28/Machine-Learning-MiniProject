"""Create a reproducible stratified subset from Airbus or labeled ship images.

Airbus input may be either:
  source/train_v2/*.jpg + source/train_ship_segmentations_v2.csv
or a labeled folder layout:
  source/ships/* and source/no_ships/*

The output layout is train/val/test/ships and train/val/test/no_ships.
"""
import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def labeled_folders(source: Path):
    folders = {
        "ship": [source / "ships", source / "ship"],
        "no_ship": [source / "no_ships", source / "no_ship"],
    }
    result = {"ship": [], "no_ship": []}
    for label, candidates in folders.items():
        for folder in candidates:
            if folder.exists():
                result[label].extend(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return result


def find_airbus_csv(source: Path):
    names = {"train_ship_segmentations_v2.csv", "train_ship_segmentations.csv"}
    candidates = [p for p in source.rglob("*.csv") if p.name in names]
    for parent in [source.parent, source.parent.parent]:
        if parent.exists():
            candidates.extend(p for p in parent.glob("*.csv") if p.name in names)
    return next(iter(dict.fromkeys(candidates)), None)


def find_airbus_images(source: Path, image_name: str, csv_path: Path):
    candidates = [
        source / image_name,
        source / "train_v2" / image_name,
        source / "train" / image_name,
        csv_path.parent / image_name,
        csv_path.parent / "train_v2" / image_name,
        csv_path.parent / "train" / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def airbus_folders(source: Path):
    csv_path = find_airbus_csv(source)
    if csv_path is None:
        return None
    labels = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        image_key = "ImageId" if "ImageId" in fields else fields[0]
        mask_key = "EncodedPixels" if "EncodedPixels" in fields else fields[1]
        for row in reader:
            image_name = row[image_key].strip()
            pixels = (row[mask_key] or "").strip()
            # Airbus has one row per object; any non-empty RLE means ship.
            labels[image_name] = labels.get(image_name, False) or bool(pixels and pixels.lower() != "nan")
    result = {"ship": [], "no_ship": []}
    for image_name, has_ship in labels.items():
        path = find_airbus_images(source, image_name, csv_path)
        if path is not None:
            result["ship" if has_ship else "no_ship"].append(path)
    return result


def unique_paths(groups):
    result = {}
    for label, paths in groups.items():
        by_hash = {}
        for path in paths:
            by_hash.setdefault(digest(path), path)
        result[label] = list(by_hash.values())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Airbus root or folder containing ships/ and no_ships/")
    ap.add_argument("--output", default="data/subset_6000")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--ship-fraction", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source, output = Path(args.source), Path(args.output)
    rng = random.Random(args.seed)
    total = args.train + args.val + args.test
    groups = airbus_folders(source) or labeled_folders(source)
    groups = unique_paths(groups)
    ship_total = round(total * args.ship_fraction)
    requested = {"ship": ship_total, "no_ship": total - ship_total}
    counts = {label: len(paths) for label, paths in groups.items()}
    if any(counts[label] < requested[label] for label in requested):
        raise SystemExit(f"Not enough unique images. Found {counts}; requested {requested}.")

    manifest = []
    for label, count in requested.items():
        paths = groups[label]
        rng.shuffle(paths)
        chosen = paths[:count]
        n_train = round(count * args.train / total)
        n_val = round(count * args.val / total)
        assignments = [("train", p) for p in chosen[:n_train]]
        assignments += [("val", p) for p in chosen[n_train:n_train + n_val]]
        assignments += [("test", p) for p in chosen[n_train + n_val:]]
        for split, path in assignments:
            destination = output / split / ("ships" if label == "ship" else "no_ships") / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            manifest.append((split, label, str(path), str(destination)))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "source", "subset_path"])
        writer.writerows(sorted(manifest))
    print(f"Created {len(manifest)} images in {output} using seed {args.seed}.")
    for split in ("train", "val", "test"):
        print(split, {label: sum(s == split and y == label for s, y, _, _ in manifest) for label in ("ship", "no_ship")})


if __name__ == "__main__":
    main()