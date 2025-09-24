# FretboardFlow

FretboardFlow is a research project on chord voicing prediction for guitar.
It introduces:

* **The FretboardFlow dataset** – expert-recorded rhythm guitar performances with multiple chord voicing variations.
* **Dual-model architectures** – combining harmonic context (chord symbols) with ergonomic context (voicing history) to predict convenient chord voicings.

This repository contains dataset utilities, model implementations, and training scripts.

it also contains a python implementation of the 2014 "[Kernel additive modeling for interference reduction in
multi-channel music recordings](https://inria.hal.science/hal-01116686v2/document)" by Thomas Prätzlich, Rachel Bittner, Antoine Liutkus, Meinard Müller

---

## Structure

```
FretboardFlow/
│
├── data/                # dataset handling
│   ├── lab_files/       # McGill Billboard chord files
│   ├── midi/            # FretboardFlow dataset
│   ├── dataset.py
│   ├── augmented_dataset.py
│   ├── dadagp.py
│   ├── fullseq_dataset.py
│   └── ...              # stats and augmented variants
│
├── models/              # model definitions
│   ├── anatomical.py    # anatomical scoring functions
│   ├── base_ff_model.py # base dual-model structure
│   ├── bi_lstm.py
│   ├── gru.py
│   ├── losses.py
│   ├── metrics.py
│   └── config.py
│
└── 
```

---

## Installation

Dependencies:

```bash
pip install -r requirements.txt   # (will be added after ISMIR)
```

---

## Usage

Example training run with 5-fold cross-validation:

```bash
python main.py \
  --k_folds 5 \
  --model dhooge \
  --dataset dadagp_full_aug \
  --history_len 1
```

Arguments:
(to be updated after ISMIR, my apologies)
---

## Dataset

* 97 recordings across 35 songs.
* Up to 5 versions of a song.
* Captured with a **hexaphonic pickup** to track string–fret positions precisely.
* Balanced fretboard coverage (up to the 10th fret).

---

## Citation

If you use this work, please cite:

```
@inproceedings{velez2025fretboardflow,
  title={FretboardFlow: A Dual-Model Approach to Optimize Chord Voicings on the Guitar Fretboard},
  author={Vélez Vásquez, Marcel A. and Baelemans, Mariëlle and Driedger, Jonathan and Burgoyne, John Ashley},
  booktitle={Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2025},
  address={Daejeon, South Korea}
}
```

---

## Status

This repo is actively evolving.

* More documentation and installation details will be added **after ISMIR**.
