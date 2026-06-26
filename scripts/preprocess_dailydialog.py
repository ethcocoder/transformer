#!/usr/bin/env python3
"""Convert DailyDialog zips to a clean text corpus for training.

Output format: one dialogue per line, utterances separated by ' | '.
This is compatible with the C++ dataset loader.
"""

import os
import zipfile
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DAILYDIALOG_DIR = os.path.join(ROOT, "dailydialog")
OUTPUT_DIR = os.path.join(ROOT, "data")


def extract_split(split_name):
    zip_path = os.path.join(DAILYDIALOG_DIR, f"{split_name}.zip")
    if not os.path.exists(zip_path):
        print(f"[WARN] {zip_path} not found, skipping")
        return []

    dialogues = []
    with zipfile.ZipFile(zip_path) as z:
        # Find the utterances file
        utt_file = None
        for f in z.namelist():
            if "dialog" in f.lower() and "act" not in f.lower() and "emotion" not in f.lower():
                utt_file = f
                break

        if not utt_file:
            print(f"[WARN] No dialogue file found in {zip_path}")
            return []

        content = z.read(utt_file).decode("utf-8")
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Split by __eou__ and filter empty
            utts = [
                u.strip() for u in line.split("__eou__") if u.strip()
            ]
            if utts:
                # Store as pipe-separated string (one dialogue per line)
                dialogues.append(" | ".join(utts))

    print(f"[PREPROCESS] {split_name}: {len(dialogues)} dialogues")
    return dialogues


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dialogues = []
    for split in ["train", "validation", "test"]:
        dialogs = extract_split(split)
        all_dialogues.extend(dialogs)

    output_path = os.path.join(OUTPUT_DIR, "dailydialog.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for d in all_dialogues:
            f.write(d + "\n")

    print(f"[PREPROCESS] Written {len(all_dialogues)} dialogues to {output_path}")
    print(f"[PREPROCESS] Total size: {os.path.getsize(output_path):,} bytes")


if __name__ == "__main__":
    main()
