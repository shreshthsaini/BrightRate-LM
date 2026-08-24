# BrightRate-LM

BrightRate-LM assesses the perceptual quality of user-generated HDR video from a multi-exposure frame stack. It returns a 0 to 100 score, a short description of visible defects, and reasoning for the score. This work extends BrightRate (WACV 2026) with a multimodal language model and a controlled study of HDR input representations.

[Paper](https://arxiv.org/abs/TODO) | [Project page](https://shreshthsaini.github.io/BrightRate-LM/) | [Primary adapter](https://huggingface.co/shreshthsaini/brightrate-lm-7b-multiexposure) | [Models](https://huggingface.co/shreshthsaini) | [BrightRate paper](https://openaccess.thecvf.com/content/WACV2026/papers/Saini_BrightRate_Quality_Assessment_for_User-Generated_HDR_Videos_WACV_2026_paper.pdf) | [BrightVQ repository](https://github.com/shreshthsaini/BrightVQ) | [CHUG](https://shreshthsaini.github.io/CHUG/) | [Beyond8Bits](https://shreshthsaini.github.io/Beyond8Bits/)

## Results

BrightRate-LM numbers are means across five content-separated 80/20 splits. The BrightRate row contains the published 100-split medians.

| Model | Input | SROCC | PLCC | KRCC | RMSE |
|---|---|---:|---:|---:|---:|
| BrightRate-LM, Qwen2.5-VL-7B | Multi-exposure | 0.9052 | 0.9107 | 0.7281 | 5.5348 |
| BrightRate, published | HDR-aware features | 0.8887 | 0.8970 | 0.7059 | 5.7514 |

## Quickstart

The demo needs Python 3.11, ffmpeg, and a CUDA GPU with about 20 GiB of free memory.

```bash
git clone https://github.com/shreshthsaini/BrightRate-LM.git
cd BrightRate-LM
uv venv --python 3.11
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/hf download shreshthsaini/brightrate-lm-7b-multiexposure \
  adapter_config.json adapter_model.safetensors \
  --local-dir adapters/brightrate-lm-7b
.venv/bin/python src/predict.py \
  --video x.mp4 \
  --adapter adapters/brightrate-lm-7b
```

The final command renders eight temporal samples at -2, 0, and +2 stops, then prints JSON with `score`, `level`, `description`, and `reasoning`.

## Training

Download `BrightVQ.csv` from the [BrightVQ repository](https://github.com/shreshthsaini/BrightVQ), place the BrightVQ videos under `data/videos/`, and render the cache:

```bash
.venv/bin/python src/build_expstack_cache.py \
  --videos-dir data/videos \
  --output-dir cache/frames8-expstack \
  --workers 8
```

Run one final split with the paper recipe:

```bash
.venv/bin/python src/train_v2.py \
  --mode final \
  --split 0 \
  --run-name brightrate-lm-split0 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --csv data/BrightVQ.csv \
  --splits configs/splits.json \
  --frames-root cache/frames8-expstack \
  --epochs 2 \
  --schedule-epochs 3 \
  --seed 700 \
  --eval-every-updates 50
```

All default paths and recipe values are in `configs/default.yaml`. Any path can also be given on the command line.

## Citation

```bibtex
@article{saini2026brightratelm,
  title   = {BrightRate-LM: Representation-Aware Quality Assessment for User-Generated HDR Video},
  author  = {Saini, Shreshth and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C.},
  journal = {Machine Vision and Applications},
  year    = {2026},
  note    = {Submitted}
}

@inproceedings{saini2026brightrate,
  title     = {BrightRate: Quality Assessment for User-Generated HDR Videos},
  author    = {Saini, Shreshth and Chen, Bowen and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C.},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages     = {1522--1532},
  year      = {2026}
}

@inproceedings{saini2025chug,
  title     = {CHUG: Crowdsourced User-Generated HDR Video Quality Dataset},
  author    = {Saini, Shreshth and Bovik, Alan C. and Birkbeck, Neil and Wang, Yilin and Adsumilli, Balu},
  booktitle = {IEEE International Conference on Image Processing},
  pages     = {2504--2509},
  year      = {2025},
  doi       = {10.1109/ICIP55913.2025.11084488}
}

@inproceedings{saini2026beyond8bits,
  title     = {Seeing Beyond 8bits: Subjective and Objective Quality Assessment of HDR-UGC Videos},
  author    = {Saini, Shreshth and Chen, Bowen and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {15538--15549},
  year      = {2026}
}
```

The code is released under the MIT License. BrightVQ data remains under its original terms.
