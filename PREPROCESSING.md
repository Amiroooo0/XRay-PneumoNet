# Chest X-ray preprocessing

Install Pillow with `python -m pip install Pillow`, then run:

```powershell
python prepare_data.py
```

This audits JPEG readability, dimensions, modes, and file sizes. It writes
`data_audit.csv` and a deterministic `split_manifest.csv`. The manifest merges
the existing training and validation folders into development data and assigns
20% of each class to validation. The test folder is excluded from that split.
This is a file-level stratified split because patient identifiers are not
available in the folder structure; if patient IDs can be recovered, split by
patient to prevent images from the same patient appearing in both sets.

To make resized inputs in a separate directory:

```powershell
python prepare_data.py --export prepared_224 --include-test
```

The export converts each image to 8-bit grayscale, corrects EXIF orientation,
fits it inside a 224 by 224 square with preserved aspect ratio, and pads the
remaining space with black pixels by default. Use `--padding-mode edge` to
fill padding with the median intensity around each resized image's border.
Use `--resize-mode crop` to center-crop a square instead; inspect those crops
for cut-off lung edges before training. Source JPEGs are never overwritten.
The optional test export keeps the existing test membership under the output's
`test` folder.

The two available padded exports are `prepared_224` (black fill) and
`prepared_edge_224` (median border-intensity fill). A visual comparison of
representative narrow, typical, and wide images is saved as
`preprocessing_comparison.png`. Center crops were not selected because the
widest examples lose substantial anatomy at the sides.

To reproduce the edge-filled version:

```powershell
python prepare_data.py --export prepared_edge_224 --include-test --resize-mode pad --padding-mode edge
```

When training, apply any random augmentation only to the training split. Keep
augmentation mild (small rotation/translation and modest brightness/contrast
variation), and use the same deterministic resizing and normalization for
validation and test. Do not use horizontal flips: laterality and asymmetric
findings can matter. Compute intensity normalization statistics from training
images only.

The source tree contains `.DS_Store` files; the utility ignores them by
selecting `.jpg` and `.jpeg` files only. The provided directory currently has
5,856 readable JPEGs (train: 5,216; val: 16; test: 624), not the README's
stated 5,863. The labels encode only `NORMAL` and `PNEUMONIA`; they do not
distinguish bacterial from viral pneumonia.
