"""Build a small Airbus subset directly from airbus-ship-detection.zip.

This avoids extracting the full ~30 GB archive. It reads the segmentation CSV,
selects a stratified ship/no-ship sample, and extracts only those images.
"""
import argparse
import csv
import hashlib
import io
import random
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to airbus-ship-detection.zip")
    ap.add_argument("--output", default="data/subset_6000")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--ship-fraction", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    output = Path(args.output)
    total = args.train + args.val + args.test
    rng = random.Random(args.seed)

    with zipfile.ZipFile(args.zip, "r") as archive:
        csv_name = next((name for name in archive.namelist() if name.endswith("train_ship_segmentations_v2.csv") or name.endswith("train_ship_segmentations.csv")), None)
        if csv_name is None:
            raise SystemExit("Could not find train_ship_segmentations_v2.csv in the ZIP.")

        labels = {}
        with archive.open(csv_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig")
            reader = csv.DictReader(text)
            for row in reader:
                image_name = row.get("ImageId") or row.get(reader.fieldnames[0])
                encoded = (row.get("EncodedPixels") or row.get(reader.fieldnames[1]) or "").strip()
                labels[image_name] = labels.get(image_name, False) or bool(encoded and encoded.lower() != "nan")

        names = set(archive.namelist())
        image_paths = {}
        for image_name in labels:
            for candidate in (image_name, "train_v2/" + image_name, "train/" + image_name):
                if candidate in names:
                    image_paths[image_name] = candidate
                    break
        groups = {"ship": [n for n, is_ship in labels.items() if is_ship and n in image_paths],
                  "no_ship": [n for n, is_ship in labels.items() if not is_ship and n in image_paths]}
        requested = {"ship": round(total * args.ship_fraction), "no_ship": total - round(total * args.ship_fraction)}
        counts = {label: len(items) for label, items in groups.items()}
        if any(counts[label] < requested[label] for label in requested):
            raise SystemExit(f"Not enough images. Found {counts}; requested {requested}.")

        manifest = []
        for label, count in requested.items():
            chosen = groups[label]
            rng.shuffle(chosen)
            chosen = chosen[:count]
            n_train = round(count * args.train / total)
            n_val = round(count * args.val / total)
            assignments = [("train", n) for n in chosen[:n_train]]
            assignments += [("val", n) for n in chosen[n_train:n_train + n_val]]
            assignments += [("test", n) for n in chosen[n_train + n_val:]]
            for split, image_name in assignments:
                destination = output / split / ("ships" if label == "ship" else "no_ships") / image_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = archive.read(image_paths[image_name])
                destination.write_bytes(data)
                manifest.append((split, label, image_paths[image_name], str(destination), hashlib.sha1(data).hexdigest()))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "archive_path", "subset_path", "sha1"])
        writer.writerows(sorted(manifest))
    print(f"Extracted {len(manifest)} images into {output} without unpacking the full archive.")
    for split in ("train", "val", "test"):
        print(split, {label: sum(s == split and y == label for s, y, _, _, _ in manifest) for label in ("ship", "no_ship")})


if __name__ == "__main__":
    main()