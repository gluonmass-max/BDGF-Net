# BDGF-Net

> The code will be released after the paper is published and indexed.

PyTorch implementation of BDGF-Net (Band-Decoupled Guided Fusion Network) for semantic segmentation of Sentinel-1 SAR and Sentinel-2 multispectral imagery, together with the re-implemented fusion baselines.

## Installation

Requires Linux, an NVIDIA GPU (CUDA 12.8 wheels), Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> bdgf-net && cd bdgf-net
uv sync
export MS_ROOT=/path/to/workdir   # datasets, pretrained weights and caches
source env.sh
```

ImageNet-pretrained MiT-B2 weights:

```bash
mkdir -p $MS_ROOT/pretrained
curl -L -o $MS_ROOT/pretrained/mit_b2.imagenet.safetensors \
  https://huggingface.co/smp-hub/mit_b2.imagenet/resolve/main/model.safetensors
```

## Data

Download Hunan, CerraData-4MM and DFC2020 from their official releases and arrange them as:

```
$MS_ROOT/datasets/
├── hunan/raw/Hunan_Dataset/{train,val,test}/{s1,s2,lc}/
├── cerradata4mm/raw/cerradata_4mm/{msi_images,sar_images,semantic_7c}/
└── dfc2020/raw/{s1,s2,dfc}_{test,validation}/
```

The DFC2020 archives extract without sub-folders; sort them with:

```bash
uv run python scripts/organize_dfc2020.py --raw $MS_ROOT/datasets/dfc2020/raw
```

Data splits (`splits/`) and normalization statistics (`stats/`) are included. To regenerate them:

```bash
uv run python scripts/index_cerradata.py --root $MS_ROOT/datasets/cerradata4mm/raw/cerradata_4mm
uv run python scripts/make_cerradata_split.py
uv run python scripts/make_dfc2020_split.py --raw $MS_ROOT/datasets/dfc2020/raw
uv run python scripts/compute_stats.py --dataset dfc2020 --data-root $MS_ROOT/datasets   # per dataset
```

Optional: `uv run python scripts/build_cache.py` stores decoded arrays in `$MS_ROOT/cache` for faster loading.

## Training and evaluation

```bash
uv run python -u scripts/train.py --dataset dfc2020 --model stitch+auxf+ffbd@ohem --seed 0
```

Each run trains with the dataset's fixed schedule, keeps the checkpoint with the best validation mIoU and evaluates it on the test split. Outputs go to `runs/<dataset>/<model>/seed<k>/` (`config.json`, `best.pt`, `val_metrics.json`, `test_metrics.json`); finished runs are skipped.

`--dataset`: `hunan`, `cerradata4mm`, `dfc2020`.

| `--model`                                                    | Method                                     |
| ------------------------------------------------------------ | ------------------------------------------ |
| `stitch+auxf+ffbd@ohem`                                      | BDGF-Net                                   |
| `s1`, `s2`, `concat`                                         | Single-modality and early-fusion baselines |
| `cmx`, `cmnext`, `tokenfusion`, `gemini`, `asanet`, `stitch` | Fusion baselines                           |

Model names follow `<base>+<module>...@<loss>~<bands>`:

- `ffbd`: BD-FreqFusion decoder; `freqfusion`: FreqFusion without guidance; `ffbdo` / `ffbds` / `ffbdx`: optical-only, SAR-only and swapped guidance
- `auxf`: fused-feature deep supervision
- `@ohem`: online hard example mining (default: cross-entropy)
- `~nore`, `~noswir`, `~rgbn`, `~vv`, `~vh`: zero an input band group on all splits (no red edge, no SWIR, RGB+NIR only, VV only, VH only)

Mean ± std over seeds:

```bash
uv run python scripts/summarize_runs.py --runs-dir runs
```

## Tests

```bash
uv run pytest
```

## Repository layout

```
msfuse/
├── data/          # dataset readers, splits, normalization, band masks
├── models/        # MiT encoder, StitchFusion, baselines, segmentor
│   └── plugins/   # BD-FreqFusion (mdec.py), FreqFusion, deep supervision (aux.py)
├── recipes.py     # losses (cross-entropy, OHEM)
├── train.py       # training loop and evaluation
└── metrics.py
scripts/           # data preparation, training, summaries
splits/  stats/    # data splits and normalization statistics
tests/
```

## Acknowledgements

This code builds on SegFormer, StitchFusion, FreqFusion, CMX, CMNeXt, TokenFusion, GeminiFusion and ASANet. We thank the authors for releasing their work.

## Citation

To be added after publication.
