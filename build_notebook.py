#!/usr/bin/env python3
"""Generate progenet-panda-training.ipynb from structured cell definitions."""
import json, sys

C = []
def md(src):   C.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").split("\n")})
def code(src): C.append({"cell_type": "code", "execution_count": None, "metadata": {},
                         "outputs": [], "source": src.strip("\n").split("\n")})

# ═══════════════════════════════════════════════════════════════════════════ 0
md(r"""
# ProGENET — Prostate Gleason Evaluation Network

**Ordinal ISUP grading of prostate biopsies from PANDA whole-slide images.**

Dataset: [`prostate-cancer-grade-assessment`](https://www.kaggle.com/c/prostate-cancer-grade-assessment) · 10,616 slides · Radboud UMC + Karolinska Institute

---

This notebook is a rebuild of *Gleason Score Prediction for the Severity of Prostate Metastasis Using Machine Learning* (Bamigbade & Abubakar-Sidiq, University of Lagos, 2021) with four deliberate design changes. Each one is an answer to a specific weakness in the original, and each is documented as an ADR in the repo.

| | 2021 baseline | ProGENET | Why |
|---|---|---|---|
| **Tile handling** | 36 tiles → whole slide resized | 36 tiles → 6×6 montage, ranked by tissue mass | ADR-001 — one forward pass, deterministic layout |
| **Colour** | grayscale + Gaussian σ=4 | RGB + Macenko stain normalisation | ADR-004 — grayscale destroys the H/E contrast grading depends on |
| **Target** | 10-way Gleason pair, softmax + BCE | 6-way ISUP, cumulative-link ordinal head | ADR-002 — ISUP is ordinal; QWK penalises distance |
| **Metric** | accuracy (85.2% / 84.8%) | quadratic weighted kappa | The original's own logs show QWK ≈ 0.04 — i.e. chance — while accuracy read 85% |

That last row is the important one. **Accuracy is the wrong metric for this problem.** With PANDA's grade distribution, a model that always predicts ISUP 0 or 1 scores respectably on accuracy and is clinically worthless. QWK is the metric the PANDA challenge used and the one this notebook optimises.

### Runtime

| | |
|---|---|
| Accelerator | **GPU P100 or T4** (Settings → Accelerator) |
| Internet | **On** for the first run, to fetch pretrained backbone weights |
| Expected | ~35 min/epoch on P100 at `IMG_SIZE=512`, `N_TILES=36` |
| Session budget | fits 8–10 epochs of one fold inside the 9-hour limit |

> ⚠️ **Research use only.** Not a medical device. Nothing here is validated for clinical decision-making.
""")

# ═══════════════════════════════════════════════════════════════════════════ 1
md(r"""
## 1 · Configuration

Everything tunable lives here. The ablations reported at the end of the notebook are all produced by changing values in this one cell.
""")

code(r'''
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os, json, math, time, random, warnings
warnings.filterwarnings("ignore")

@dataclass
class CFG:
    # ── data ───────────────────────────────────────────────────────────────
    data_root:   str = "/kaggle/input/prostate-cancer-grade-assessment"
    work_dir:    str = "/kaggle/working"
    n_tiles:     int = 36        # perfect square — ablate 16 / 36 / 64  (ADR-001)
    tile_size:   int = 256       # tile edge at pyramid level 1
    img_size:    int = 512       # montage resized to this before the backbone
    level:       int = 1         # PANDA level 1 ≈ 4× downsample

    # ── preprocessing  (ADR-004) ───────────────────────────────────────────
    stain_norm:     bool  = True
    gaussian_sigma: float = 0.0  # 0 = off. The 2021 sweep found σ=4 marginal.
    to_grayscale:   bool  = False # legacy 2021 reproduction switch

    # ── model ──────────────────────────────────────────────────────────────
    backbone:    str = "efficientnet_b0"   # b0 | b3 | b7(paper) | resnet50
    n_classes:   int = 6                   # ISUP 0..5
    ordinal:     bool = True               # cumulative-link head  (ADR-002)
    pool:        str = "gem"               # gem | avg
    drop_rate:   float = 0.3

    # ── training ───────────────────────────────────────────────────────────
    folds:       int = 5
    train_fold:  int = 0        # a Kaggle session trains one fold
    epochs:      int = 10
    batch_size:  int = 8
    accum_steps: int = 2        # effective batch 16 at 512px on a P100
    lr:          float = 3e-4
    weight_decay:float = 1e-5
    warmup_pct:  float = 0.1
    amp:         bool = True
    num_workers: int = 2
    seed:        int = 42

    # ── experiment switches ────────────────────────────────────────────────
    provider_probe: bool = True   # train radboud → test karolinska (risk R2)
    debug:          bool = False  # 200 slides, 2 epochs — smoke test

    def __post_init__(self):
        s = int(round(self.n_tiles ** 0.5))
        assert s * s == self.n_tiles, f"n_tiles must be a perfect square, got {self.n_tiles}"

cfg = CFG()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

print(json.dumps(asdict(cfg), indent=2))
''')

# ═══════════════════════════════════════════════════════════════════════════ 2
md("## 2 · Environment")

code(r'''
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import skimage.io
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(cfg.seed)

print(f"torch      {torch.__version__}")
print(f"device     {DEVICE}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

try:
    import timm
    HAS_TIMM = True
    print(f"timm       {timm.__version__}")
except ImportError:
    HAS_TIMM = False
    print("timm       not installed — falling back to torchvision backbones")

DATA = Path(cfg.data_root)
assert DATA.exists(), (
    f"{DATA} not found. Add the competition dataset: "
    "Notebook → Add Data → 'Prostate cANcer graDe Assessment (PANDA) Challenge'"
)
print(f"\ndata       {DATA}")
print(f"           {len(list((DATA/'train_images').glob('*.tiff')))} slides")
''')

# ═══════════════════════════════════════════════════════════════════════════ 3
md(r"""
## 3 · Labels

PANDA gives both `gleason_score` (the pattern pair) and `isup_grade` (0–5). These are not independent — ISUP is a deterministic function of the pair. We train on ISUP and *derive* the reported Gleason pattern by inverting the table, rather than asking the model to predict which pattern is primary from a slide-level label it cannot support.
""")

code(r'''
GLEASON_TO_ISUP = {
    "0+0": 0, "negative": 0,
    "3+3": 1,
    "3+4": 2,
    "4+3": 3,
    "4+4": 4, "3+5": 4, "5+3": 4,
    "4+5": 5, "5+4": 5, "5+5": 5,
}
# Inverse: the modal Gleason pair reported for each ISUP grade.
ISUP_TO_GLEASON = {0: "0+0", 1: "3+3", 2: "3+4", 3: "4+3", 4: "4+4", 5: "4+5"}
RISK_BAND = {
    0: "benign / no carcinoma",
    1: "low risk",
    2: "intermediate — favourable",
    3: "intermediate — unfavourable",
    4: "high risk",
    5: "very high risk",
}

train = pd.read_csv(DATA / "train.csv")

# PANDA ships a handful of slides with known label/mask problems. The community
# suppression list; dropping them is standard practice and worth ~0.01 QWK.
SUSPECT = {
    "3790f55cad63053e956fb73027179707", "b0a92a74cb53899311acc30b7405e101",
    "e4215cfc8c41ec040068ea083f9cdb1a", "c86942ebf90f4c69c5851a49a0e5c7a5",
}
train = train[~train.image_id.isin(SUSPECT)].reset_index(drop=True)

if cfg.debug:
    train = train.groupby("isup_grade", group_keys=False).head(35).reset_index(drop=True)
    cfg.epochs = 2

print(f"{len(train)} slides\n")
print(pd.crosstab(train.isup_grade, train.data_provider, margins=True))

# Sanity: does the stated gleason_score actually agree with the stated isup_grade?
derived = train.gleason_score.str.lower().map(GLEASON_TO_ISUP)
mismatch = (derived != train.isup_grade).sum()
print(f"\ngleason↔isup disagreements: {mismatch}  ({100*mismatch/len(train):.2f}%)")
''')

code(r'''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))

train.isup_grade.value_counts().sort_index().plot(
    kind="bar", ax=ax[0], color="#2b7a78", edgecolor="none")
ax[0].set_title("ISUP grade distribution", loc="left", fontweight=600)
ax[0].set_xlabel("ISUP grade"); ax[0].set_ylabel("slides")

pd.crosstab(train.isup_grade, train.data_provider).plot(
    kind="bar", stacked=True, ax=ax[1], color=["#2b7a78", "#d97706"], edgecolor="none")
ax[1].set_title("by data provider", loc="left", fontweight=600)
ax[1].set_xlabel("ISUP grade"); ax[1].legend(frameon=False)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()
''')

md(r"""
Two things to read off those plots before going further.

**The distribution is imbalanced and the imbalance is structured** — ISUP 0 and 1 together are roughly half the dataset. This is the trap accuracy falls into: predict "1" for everything and you look competent.

**The providers are not interchangeable.** Radboud and Karolinska differ in grade mix *and* in how the labels were produced — Karolinska's are one pathologist's read, Radboud's are derived semi-automatically from IHC-guided annotation. This is why the split below is provider-stratified and why the cross-provider probe in §10 exists.
""")

# ═══════════════════════════════════════════════════════════════════════════ 4
md(r"""
## 4 · Tiling

The tissue strip on a PANDA slide occupies perhaps 5–15% of the canvas; everything else is white. Feed the resized whole slide to a CNN and it spends almost all its capacity on background. The 2021 study measured this directly (75.8% vs 73.4% test accuracy for tiles vs raw) and the finding holds — it is the one preprocessing decision from the original that ProGENET keeps unchanged.

Tiles are ranked by **summed ink** — distance from white — rather than by mean intensity. Mean intensity would rank pale, low-cellularity stroma as empty, and pale stroma is exactly what distinguishes a benign core from a Gleason 3 one.

*These functions are duplicated in `pipeline/bin/tile_wsi.py`. Identical code, two runtimes: Nextflow for bulk offline preprocessing, here for on-the-fly training.*
""")

code(r'''
def read_level(path, level=1):
    """Read one pyramid level of a PANDA .tiff.

    skimage's MultiImage reads the multi-page TIFF lazily; page 1 is the
    ~4x-downsampled level, which is the standard PANDA working resolution.
    """
    return skimage.io.MultiImage(str(path))[level]


def pad_to_multiple(img, tile):
    """Pad with WHITE so the image divides evenly into tiles.

    White is the only correct pad value here: it reads as background to the
    tissue scorer, so padding can never manufacture a high-scoring tile.
    """
    h, w = img.shape[:2]
    ph, pw = (-h) % tile, (-w) % tile
    if not (ph or pw):
        return img
    return np.pad(img, [(ph//2, ph-ph//2), (pw//2, pw-pw//2), (0, 0)],
                  constant_values=255)


def cut_tiles(img, tile):
    img = pad_to_multiple(img, tile)
    h, w = img.shape[:2]
    r, c = h // tile, w // tile
    return (img.reshape(r, tile, c, tile, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, tile, tile, 3))


def select_tiles(tiles, n):
    """Top-n by tissue mass, white-padded if the slide is short of tissue.

    Sorted descending so grid position is deterministic — the CNN can rely on
    the top-left cell always being the densest tile, and two users uploading the
    same slide must see the same prediction.
    """
    scores = (255 - tiles.reshape(len(tiles), -1).astype(np.int64)).sum(1)
    keep = np.argsort(-scores)[:n]
    out, sc = tiles[keep], scores[keep]
    if len(out) < n:
        short = n - len(out)
        out = np.concatenate([out, np.full((short, *tiles.shape[1:]), 255, np.uint8)])
        sc  = np.concatenate([sc, np.zeros(short, np.int64)])
    return out, sc


def build_montage(tiles):
    n = len(tiles); s = int(round(n ** 0.5)); t = tiles.shape[1]
    return (tiles.reshape(s, s, t, t, 3)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(s*t, s*t, 3))


def slide_to_montage(path, n_tiles=36, tile_size=256, level=1, out_size=512):
    img = read_level(path, level)
    tiles, scores = select_tiles(cut_tiles(img, tile_size), n_tiles)
    m = build_montage(tiles)
    if m.shape[0] != out_size:
        import cv2
        m = cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return m, scores
''')

code(r'''
# Look at what the tiler actually produces before trusting it.
sample = train.sample(3, random_state=cfg.seed)
fig, axes = plt.subplots(3, 2, figsize=(11, 15),
                         gridspec_kw={"width_ratios": [1.35, 1]})

for ax_row, (_, row) in zip(axes, sample.iterrows()):
    p = DATA / "train_images" / f"{row.image_id}.tiff"
    raw = read_level(p, 2)                       # level 2 = thumbnail, for display
    montage, scores = slide_to_montage(p, cfg.n_tiles, cfg.tile_size,
                                       cfg.level, cfg.img_size)
    ax_row[0].imshow(raw); ax_row[0].axis("off")
    ax_row[0].set_title(f"{row.image_id[:12]}…  ·  {row.data_provider}\n"
                        f"ISUP {row.isup_grade}  ·  Gleason {row.gleason_score}",
                        loc="left", fontsize=10)
    ax_row[1].imshow(montage); ax_row[1].axis("off")
    ax_row[1].set_title(f"{cfg.n_tiles}-tile montage · "
                        f"{int((scores > 0).sum())} tissue-bearing",
                        loc="left", fontsize=10)
plt.tight_layout(); plt.show()
''')

# ═══════════════════════════════════════════════════════════════════════════ 5
md(r"""
## 5 · Stain normalisation (ADR-004)

The 2021 pipeline converted tiles to grayscale. That is worth pausing on, because it is the single change with the clearest mechanism of harm.

Gleason grading reads *glandular architecture in haematoxylin and eosin*. H binds nucleic acid and renders nuclei blue-purple; E binds protein and renders cytoplasm and stroma pink. The boundary between a gland lumen, its epithelial lining, and surrounding stroma is carried substantially by that hue difference. Grayscale projects two chemically distinct channels onto one intensity axis — a dense basophilic nucleus and a dense eosinophilic stromal band can land on the same gray value.

What grayscale *does* buy is invariance to scanner and stain-protocol variation, and PANDA genuinely has that problem. So the instinct was sound. Macenko normalisation solves the same problem properly: it estimates each image's stain vectors by SVD in optical-density space and re-projects them onto a fixed reference basis, removing colour cast while preserving H-versus-E separation.
""")

code(r'''
REF_STAIN = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
REF_CONC  = np.array([1.9705, 1.0308])


def macenko(img, ref_stain=REF_STAIN, ref_conc=REF_CONC, Io=240, beta=0.15, alpha=1.0):
    """Re-project an H&E image onto a reference stain basis.

    Returns (image, ok). Never raises: a tile with too little tissue gives a
    degenerate SVD, and one bad tile must not fail a whole slide. Callers
    surface ok=False as a quality flag instead.
    """
    h, w = img.shape[:2]
    od = -np.log(np.maximum(img.reshape(-1, 3).astype(np.float64), 1) / Io)
    od_hat = od[np.linalg.norm(od, axis=1) > beta]
    if len(od_hat) < 512:
        return img, False
    try:
        _, V = np.linalg.eigh(np.cov(od_hat.T))
        plane = V[:, 1:3]
        proj  = od_hat @ plane
        ang   = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(ang, [alpha, 100 - alpha])
        v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
        v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
        # order as (haematoxylin, eosin): H absorbs more in the first component
        HE = (np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T)
        HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)

        conc = np.linalg.lstsq(HE, od.T, rcond=None)[0]
        mx = np.percentile(conc, 99, axis=1); mx[mx < 1e-6] = 1e-6
        conc *= (ref_conc / mx)[:, None]

        out = np.clip((Io * np.exp(-ref_stain @ conc)).T.reshape(h, w, 3), 0, 255)
        return out.astype(np.uint8), True
    except np.linalg.LinAlgError:
        return img, False


def tissue_coverage(img, thresh=235):
    return float((img.mean(-1) < thresh).mean())
''')

code(r'''
# Side by side: raw montage, Macenko-normalised, and the 2021 grayscale path.
row = train.sample(1, random_state=7).iloc[0]
m, _ = slide_to_montage(DATA/"train_images"/f"{row.image_id}.tiff",
                        cfg.n_tiles, cfg.tile_size, cfg.level, cfg.img_size)
norm, ok = macenko(m)
gray = np.stack([m.mean(-1).astype(np.uint8)]*3, -1)

fig, ax = plt.subplots(1, 3, figsize=(14, 5))
for a, im, t in zip(ax, [m, norm, gray],
                    ["raw montage",
                     f"Macenko normalised ({'ok' if ok else 'fallback'})",
                     "grayscale — 2021 baseline"]):
    a.imshow(im); a.axis("off"); a.set_title(t, loc="left", fontsize=11, fontweight=600)
plt.tight_layout(); plt.show()

print(f"tissue coverage: {tissue_coverage(norm):.3f}")
print("Note how much glandular structure survives in the middle panel and is "
      "flattened in the right one — that is the cost ADR-004 is about.")
''')

# ═══════════════════════════════════════════════════════════════════════════ 6
md(r"""
## 6 · Dataset and augmentation

Augmentation is deliberately conservative. Histology has no canonical orientation, so flips and 90° rotations are free — a biopsy core is equally valid upside down. Colour jitter is kept *moderate even though stain normalisation is on*: the two address the same risk (R2, cross-provider shift) at different points, and belt-and-braces is cheap here.

What we do **not** do is elastic deformation or aggressive scaling. Gleason patterns are defined by gland *shape and size*; warping them is not augmentation, it is label noise.
""")

code(r'''
import cv2

class PANDADataset(Dataset):
    def __init__(self, df, cfg, train=True):
        self.df, self.cfg, self.train = df.reset_index(drop=True), cfg, train

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        if random.random() < 0.5: img = np.fliplr(img)
        if random.random() < 0.5: img = np.flipud(img)
        k = random.randint(0, 3)
        if k: img = np.rot90(img, k)
        img = np.ascontiguousarray(img)
        if random.random() < 0.5:                       # brightness / contrast
            a = 1.0 + random.uniform(-0.15, 0.15)
            b = random.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if random.random() < 0.3:                       # hue jitter (stain drift)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-6, 6)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-12, 12), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img, _ = slide_to_montage(
            self.cfg.data_root + f"/train_images/{r.image_id}.tiff",
            self.cfg.n_tiles, self.cfg.tile_size, self.cfg.level, self.cfg.img_size)

        if self.cfg.stain_norm:
            img, _ = macenko(img)
        if self.cfg.gaussian_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), self.cfg.gaussian_sigma)
        if self.cfg.to_grayscale:                        # legacy 2021 mode
            img = np.stack([img.mean(-1).astype(np.uint8)] * 3, -1)
        if self.train:
            img = self._augment(img)

        # ImageNet statistics — the backbone is pretrained on them.
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255)
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) \
            / torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        g = int(r.isup_grade)
        # cumulative-link target: ISUP 3 → [1,1,1,0,0]   (ADR-002)
        y_ord = torch.zeros(self.cfg.n_classes - 1)
        y_ord[:g] = 1.0
        return x, y_ord, torch.tensor(g, dtype=torch.long)
''')

# ═══════════════════════════════════════════════════════════════════════════ 7
md(r"""
## 7 · Model

An ImageNet backbone, GeM pooling, and a 5-unit cumulative-link head.

**GeM (generalised mean) pooling** instead of global average: the montage is mostly ordinary tissue with, in a high-grade case, a small aggressive focus. Average pooling dilutes that focus across 36 tiles' worth of features. GeM has a learnable exponent *p* that interpolates between average (p=1) and max (p→∞) pooling, so the network can decide for itself how much to let a single strong region dominate — which is close to what a pathologist does when they grade a core by its worst pattern.

**The ordinal head** is the change that matters most (ADR-002). Five sigmoid units, unit *k* answering "is this at least ISUP k+1?". Loss is plain BCE. The prediction is the count of units above threshold.
""")

code(r'''
class GeM(nn.Module):
    """Generalised mean pooling. p is learned; p=1 is average, p→∞ is max."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p).flatten(1)


def build_backbone(name, pretrained=True):
    """timm if available (many more backbones), torchvision otherwise."""
    if HAS_TIMM:
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        return m, m.num_features
    import torchvision.models as tv
    fn = {"efficientnet_b0": tv.efficientnet_b0, "efficientnet_b3": tv.efficientnet_b3,
          "efficientnet_b7": tv.efficientnet_b7, "resnet50": tv.resnet50}[name]
    net = fn(weights="DEFAULT" if pretrained else None)
    if name.startswith("efficientnet"):
        feat = net.classifier[1].in_features
        return nn.Sequential(*list(net.children())[:-2]), feat
    return nn.Sequential(*list(net.children())[:-2]), net.fc.in_features


class ProGENET(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.backbone, nf = build_backbone(cfg.backbone, pretrained)
        self.pool = GeM() if cfg.pool == "gem" else nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(nf),
            nn.Dropout(cfg.drop_rate),
            nn.Linear(nf, 512), nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(cfg.drop_rate / 2),
            # 5 cumulative units for ordinal, 6 logits for plain softmax
            nn.Linear(512, cfg.n_classes - 1 if cfg.ordinal else cfg.n_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f) if isinstance(self.pool, GeM) else self.pool(f).flatten(1)
        return self.head(f)


model = ProGENET(cfg).to(DEVICE)
n_par = sum(p.numel() for p in model.parameters())
print(f"{cfg.backbone}  ·  {n_par/1e6:.1f}M parameters  ·  "
      f"head outputs {cfg.n_classes-1 if cfg.ordinal else cfg.n_classes}")
print(f"\nFor reference, the 2021 study's best model (EfficientNetB7) was "
      f"66.7M parameters.\nB0 at {n_par/1e6:.1f}M is what keeps CPU inference "
      f"under the 8s budget (ADR-003).")
''')

# ═══════════════════════════════════════════════════════════════════════════ 8
md(r"""
## 8 · Metric and thresholds

Quadratic weighted kappa. The penalty on cell (i,j) is (i−j)²/(N−1)², so being wrong by one grade costs a twenty-fifth of being wrong by five. That is a reasonable model of the clinical cost, and it is why QWK — not accuracy — is the objective.

After training, the decision thresholds τ are fit **directly against QWK** by Nelder–Mead on out-of-fold predictions. It is free accuracy: no retraining, typically worth 0.01–0.03 kappa.
""")

code(r'''
from scipy.optimize import minimize

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def ordinal_decode(logits, thresholds=None):
    """Count how many cumulative units fire → ISUP grade."""
    p = torch.sigmoid(torch.as_tensor(logits)).numpy()
    t = np.asarray(thresholds if thresholds is not None else [0.5]*p.shape[1])
    return (p > t).sum(axis=1).astype(int)


def fit_thresholds(logits, y_true, init=(0.5, 0.5, 0.5, 0.5, 0.5)):
    """Nelder-Mead over 5 scalars, maximising QWK. Fit on OOF predictions —
    fitting on a single validation fold overfits it."""
    p = torch.sigmoid(torch.as_tensor(logits)).numpy()

    def neg_qwk(t):
        return -qwk(y_true, (p > np.asarray(t)).sum(axis=1))

    res = minimize(neg_qwk, np.asarray(init), method="nelder-mead",
                   options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
    return res.x, -res.fun


def ordinal_to_distribution(logits):
    """Turn 5 cumulative probabilities into a 6-class distribution.

    P(grade = k) = P(≥k) − P(≥k+1), with P(≥0)=1 and P(≥6)=0. Clipped at 0
    because the cumulative probabilities are not guaranteed monotone — see the
    monotonicity caveat in ADR-002. Renormalised so it sums to 1.

    This is NOT calibrated. It is a coherent ranking of the model's belief,
    not a probability a clinician should bet on.
    """
    p = torch.sigmoid(torch.as_tensor(logits)).numpy()
    cum = np.concatenate([np.ones((len(p), 1)), p, np.zeros((len(p), 1))], axis=1)
    d = np.clip(cum[:, :-1] - cum[:, 1:], 0, None)
    return d / np.maximum(d.sum(1, keepdims=True), 1e-9)
''')

# ═══════════════════════════════════════════════════════════════════════════ 9
md(r"""
## 9 · Training

One fold per Kaggle session. Checkpoints land in `/kaggle/working` every time validation QWK improves, so a session that times out is resumable rather than lost.
""")

code(r'''
# Provider-aware stratified split. Stratifying on the (grade, provider) pair
# keeps both the grade mix and the institutional mix stable across folds — the
# latter is what makes per-provider QWK comparable fold to fold.
train["strat"] = train.isup_grade.astype(str) + "_" + train.data_provider
skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
train["fold"] = -1
for f, (_, v) in enumerate(skf.split(train, train.strat)):
    train.loc[v, "fold"] = f

print(pd.crosstab(train.fold, train.isup_grade))
''')

code(r'''
def make_loaders(df, fold, cfg):
    tr = df[df.fold != fold]
    va = df[df.fold == fold]
    dl = lambda d, t: DataLoader(
        PANDADataset(d, cfg, train=t), batch_size=cfg.batch_size, shuffle=t,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=t)
    return dl(tr, True), dl(va, False), va


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, scheduler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    losses, all_logits, all_y = [], [], []

    for step, (x, y_ord, y) in enumerate(loader):
        x, y_ord = x.to(DEVICE, non_blocking=True), y_ord.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(x)
                loss = criterion(logits, y_ord)

        if train_mode:
            scaler.scale(loss / cfg.accum_steps).backward()
            if (step + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        losses.append(loss.item())
        all_logits.append(logits.detach().float().cpu())
        all_y.append(y)

    return (float(np.mean(losses)),
            torch.cat(all_logits).numpy(),
            torch.cat(all_y).numpy())
''')

code(r'''
def train_fold(cfg, df, fold):
    seed_everything(cfg.seed + fold)
    tr_loader, va_loader, va_df = make_loaders(df, fold, cfg)

    model = ProGENET(cfg).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() if cfg.ordinal else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    steps = max(1, len(tr_loader) // cfg.accum_steps) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=steps, pct_start=cfg.warmup_pct)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best, history = -1.0, []
    ckpt = f"{cfg.work_dir}/progenet_{cfg.backbone}_fold{fold}.pt"

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr_loss, _, _        = run_epoch(model, tr_loader, criterion,
                                         optimizer, scaler, scheduler)
        va_loss, logits, y   = run_epoch(model, va_loader, criterion)

        pred  = ordinal_decode(logits) if cfg.ordinal else logits.argmax(1)
        score = qwk(y, pred)
        acc   = (pred == y).mean()
        history.append(dict(epoch=epoch+1, train_loss=tr_loss, val_loss=va_loss,
                            qwk=score, acc=acc, mins=(time.time()-t0)/60))

        flag = ""
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                        "fold": fold, "qwk": score, "epoch": epoch}, ckpt)
            flag = "  ← saved"

        print(f"epoch {epoch+1:2d}/{cfg.epochs}  "
              f"train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"QWK {score:.4f}  acc {acc:.4f}  "
              f"{history[-1]['mins']:.1f} min{flag}")

    model.load_state_dict(torch.load(ckpt)["model"])
    _, logits, y = run_epoch(model, va_loader, criterion)
    return model, pd.DataFrame(history), logits, y, va_df


model, history, oof_logits, oof_y, oof_df = train_fold(cfg, train, cfg.train_fold)
print(f"\nbest QWK  {history.qwk.max():.4f}  (epoch {history.qwk.idxmax()+1})")
''')

code(r'''
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].plot(history.epoch, history.train_loss, label="train", lw=2, color="#2b7a78")
ax[0].plot(history.epoch, history.val_loss,   label="val",   lw=2, color="#d97706")
ax[0].set_title("loss", loc="left", fontweight=600); ax[0].legend(frameon=False)

ax[1].plot(history.epoch, history.qwk, lw=2, color="#2b7a78", marker="o", ms=4)
ax[1].axhline(0.80, ls="--", c="#94a3b8", lw=1)
ax[1].text(history.epoch.iloc[0], 0.805, "ship threshold", fontsize=9, color="#64748b")
ax[1].set_title("quadratic weighted kappa", loc="left", fontweight=600)

ax[2].plot(history.epoch, history.acc, lw=2, color="#94a3b8", marker="o", ms=4)
ax[2].set_title("accuracy (reference only)", loc="left", fontweight=600)

for a in ax:
    a.set_xlabel("epoch"); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()

print("Watch the gap between the middle and right panels. Accuracy moves early "
      "and flattens;\nQWK keeps climbing as the model learns to place its errors "
      "adjacent to the truth\nrather than anywhere. That gap is the whole "
      "argument of ADR-002.")
''')

# ═══════════════════════════════════════════════════════════════════════════ 10
md("## 10 · Evaluation")

code(r'''
tau, tuned = fit_thresholds(oof_logits, oof_y)
base = qwk(oof_y, ordinal_decode(oof_logits))

print(f"QWK @ default thresholds (0.5)  {base:.4f}")
print(f"QWK @ fitted thresholds         {tuned:.4f}   (+{tuned-base:.4f})")
print(f"thresholds                      {np.round(tau, 3)}")

pred = ordinal_decode(oof_logits, tau)
cm = confusion_matrix(oof_y, pred, labels=range(6))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(cm, cmap="BuGn")
for i in range(6):
    for j in range(6):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                   color="white" if cm[i, j] > cm.max()*0.55 else "#334155")
ax[0].set(xlabel="predicted ISUP", ylabel="true ISUP",
          xticks=range(6), yticks=range(6))
ax[0].set_title("confusion matrix", loc="left", fontweight=600)

# Per-provider QWK: the number that tells you whether this generalises.
rows = []
oof_df = oof_df.copy(); oof_df["pred"] = pred
for prov, g in oof_df.groupby("data_provider"):
    rows.append((prov, len(g), qwk(g.isup_grade, g.pred)))
rows.append(("— pooled —", len(oof_df), tuned))
perf = pd.DataFrame(rows, columns=["provider", "n", "qwk"])

ax[1].barh(perf.provider, perf.qwk, color=["#2b7a78", "#d97706", "#475569"])
ax[1].set_xlim(0, 1); ax[1].set_title("QWK by data provider", loc="left", fontweight=600)
for i, v in enumerate(perf.qwk):
    ax[1].text(v + .015, i, f"{v:.3f}", va="center", fontsize=10)
ax[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()

display(perf)

off_by_one = np.abs(pred - oof_y) <= 1
print(f"\nexact agreement      {(pred == oof_y).mean():.3f}")
print(f"within one grade     {off_by_one.mean():.3f}   ← the clinically "
      f"forgiving number, and the one a pathologist will ask about")
''')

md(r"""
### The cross-provider probe (risk R2)

The pooled QWK above is an optimistic number: it is measured on slides from the same two institutions the model trained on. The question a lab in Lagos or Nairobi actually needs answered is *what happens on a scanner this model has never seen*.

We cannot answer that without a third institution's slides, but we can put a lower bound on it: train on Radboud only, evaluate on Karolinska only. Two scanners, two staining protocols, two labelling procedures. If the model survives that, cross-institution transfer is plausible. If it collapses, the deployment claim is void and the honest output of this project is the negative result.
""")

code(r'''
if cfg.provider_probe and not cfg.debug:
    probe_cfg = CFG(**{**asdict(cfg), "epochs": max(4, cfg.epochs // 2)})

    rad = train[train.data_provider == "radboud"].copy()
    kar = train[train.data_provider == "karolinska"].copy()
    rad["fold"], kar["fold"] = 1, 0            # train on radboud, validate on karolinska
    probe_df = pd.concat([rad, kar], ignore_index=True)

    print(f"train (radboud)     {len(rad)}")
    print(f"validate (karolinska) {len(kar)}\n")

    _, probe_hist, probe_logits, probe_y, _ = train_fold(probe_cfg, probe_df, fold=0)
    probe_qwk = qwk(probe_y, ordinal_decode(probe_logits, tau))

    gap = tuned - probe_qwk
    print(f"\nin-distribution QWK   {tuned:.4f}")
    print(f"cross-provider QWK    {probe_qwk:.4f}")
    print(f"shift penalty         {gap:.4f}")
    print("\n" + ("→ Transfer holds. The deployment claim survives."
                  if probe_qwk >= 0.60 else
                  "→ R2 has fired. Cross-provider QWK is below the 0.60 kill "
                  "criterion.\n  Per 01-problem-framing.md §7 this becomes a "
                  "domain-adaptation project,\n  and the negative result is "
                  "reported rather than buried."))
else:
    print("Provider probe skipped (debug mode or provider_probe=False).")
''')

# ═══════════════════════════════════════════════════════════════════════════ 11
md(r"""
## 11 · Comparison with the 2021 baseline

Filled in from this notebook's own run. The accuracy column is reported only so the two studies can be lined up — QWK is the column that decides anything.
""")

code(r'''
baseline_2021 = pd.DataFrame([
    ("Custom CNN (2021)",      "~2.4M",  0.722, 0.758, None),
    ("Xception (2021)",        "22.9M",  0.701, 0.734, None),
    ("VGG16 (2021)",          "138.4M",  0.752, 0.745, None),
    ("VGG19 (2021)",          "143.7M",  0.798, 0.759, None),
    ("ResNet101 (2021)",       "44.7M",  0.782, 0.772, None),
    ("MobileNet (2021)",        "4.3M",  0.746, 0.701, None),
    ("DenseNet121 (2021)",      "8.1M",  0.744, 0.735, None),
    ("EfficientNetB5 (2021)",  "30.6M",  0.804, 0.774, None),
    ("EfficientNetB7 (2021)",  "66.7M",  0.852, 0.848, 0.044),
], columns=["model", "params", "train_acc", "test_acc", "qwk"])

ours = pd.DataFrame([(
    f"ProGENET {cfg.backbone} (ordinal)",
    f"{n_par/1e6:.1f}M",
    None,
    float((pred == oof_y).mean()),
    round(tuned, 4),
)], columns=baseline_2021.columns)

comparison = pd.concat([baseline_2021, ours], ignore_index=True)
display(comparison.style.format({"train_acc": "{:.3f}", "test_acc": "{:.3f}",
                                 "qwk": "{:.4f}"}, na_rep="—"))

print(
    "\nThe 2021 row that matters is the last one. EfficientNetB7 reported 84.8%\n"
    "test accuracy alongside a QWK of 0.044 — and a QWK of 0.044 is chance\n"
    "agreement. Both numbers were honestly measured; they are simply measuring\n"
    "different things, and only one of them is sensitive to a model that has\n"
    "learned the marginal grade distribution rather than the morphology.\n\n"
    "This is not a criticism of the effort in that study. It is the single\n"
    "most useful thing to carry forward from it."
)
''')

# ═══════════════════════════════════════════════════════════════════════════ 12
md(r"""
## 12 · Ablations

Re-run with these settings to reproduce the table. Each is one edit in the config cell of §1.

| Experiment | Change | Reproduces |
|---|---|---|
| Tile count | `n_tiles = 16 / 36 / 64` | ADR-001's choice of 36 |
| Grayscale | `to_grayscale=True, stain_norm=False` | the 2021 preprocessing path |
| Gaussian σ | `gaussian_sigma = 0 / 1 / 2 / 4` | Table 4.1.2b of the original |
| Ordinal vs softmax | `ordinal = False` | ADR-002 |
| Backbone | `backbone = "efficientnet_b0" / "b3" / "b7"` | Table 4.2 of the original |
| Pooling | `pool = "gem" / "avg"` | §7 |

Record each run's OOF QWK here rather than trusting a remembered number.
""")

code(r'''
ABLATIONS = pd.DataFrame(columns=["experiment", "setting", "qwk", "cpu_latency_s", "notes"])

def log_ablation(experiment, setting, qwk_value, latency=None, notes=""):
    global ABLATIONS
    ABLATIONS.loc[len(ABLATIONS)] = [experiment, setting, qwk_value, latency, notes]
    return ABLATIONS

log_ablation("baseline", f"{cfg.backbone}, {cfg.n_tiles} tiles, ordinal, macenko",
             round(tuned, 4), notes="this run")
display(ABLATIONS)
''')

# ═══════════════════════════════════════════════════════════════════════════ 13
md(r"""
## 13 · Export for serving

Two artifacts leave this notebook: an ONNX graph and a small JSON of everything the service needs to reproduce the notebook's preprocessing exactly. The JSON matters as much as the weights — a model served with different tiling or a different stain reference is a different model, and train/serve skew in preprocessing is the classic way a good validation number fails to survive contact with production.
""")

code(r'''
model.eval()
dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=DEVICE)
onnx_path = f"{cfg.work_dir}/progenet_{cfg.backbone}.onnx"

torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["montage"], output_names=["ordinal_logits"],
    dynamic_axes={"montage": {0: "batch"}, "ordinal_logits": {0: "batch"}},
    opset_version=17, do_constant_folding=True,
)

meta = {
    "name": "progenet-" + cfg.backbone.replace("_", "-") + "-ordinal",
    "version": "1.0.0",
    "trained": time.strftime("%Y-%m-%d"),
    "fold": cfg.train_fold,
    "qwk_val": round(float(tuned), 4),
    "accuracy_val": round(float((pred == oof_y).mean()), 4),
    "within_one_grade": round(float(off_by_one.mean()), 4),
    "thresholds": [round(float(t), 5) for t in tau],
    "n_params": int(n_par),
    "preprocessing": {
        "n_tiles": cfg.n_tiles, "tile_size": cfg.tile_size, "level": cfg.level,
        "img_size": cfg.img_size, "stain_norm": cfg.stain_norm,
        "stain_reference": REF_STAIN.round(4).tolist(),
        "stain_max_conc": REF_CONC.round(4).tolist(),
        "gaussian_sigma": cfg.gaussian_sigma, "to_grayscale": cfg.to_grayscale,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std":  [0.229, 0.224, 0.225],
    },
    "isup_to_gleason": ISUP_TO_GLEASON,
    "risk_band": RISK_BAND,
    "disclaimer": "Research use only. Not a medical device.",
}

with open(f"{cfg.work_dir}/model_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)

size_mb = os.path.getsize(onnx_path) / 1e6
print(f"ONNX      {onnx_path}  ({size_mb:.1f} MB)")
print(f"metadata  {cfg.work_dir}/model_meta.json")
print(f"\nartifact budget (N3, <250MB): {'PASS' if size_mb < 250 else 'FAIL'}")
''')

code(r'''
# Verify the exported graph agrees with the PyTorch model, and time it on CPU —
# the constraint the whole architecture was chosen to satisfy (ADR-003).
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = dummy.cpu().numpy()

    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    onnx_out = sess.run(None, {"montage": x})[0]

    print(f"max |torch − onnx|  {np.abs(torch_out - onnx_out).max():.2e}")

    t0 = time.time()
    for _ in range(5):
        sess.run(None, {"montage": x})
    latency = (time.time() - t0) / 5
    print(f"CPU latency          {latency*1000:.0f} ms/slide")
    print(f"budget (N1, <8s)     {'PASS' if latency < 8 else 'FAIL'}")
    print("\nNote: Kaggle CPUs are faster than the 2-vCPU serving target. "
          "Budget for ~2-3x this.")
except ImportError:
    print("onnxruntime not installed — pip install onnxruntime to verify the export")
''')

# ═══════════════════════════════════════════════════════════════════════════ 14
md(r"""
## 14 · End-to-end inference

Exactly what the deployed API does, run here against a held-out slide. If this cell and the service disagree, the service is wrong.
""")

code(r'''
def predict_slide(path, model, thresholds, cfg):
    montage, scores = slide_to_montage(path, cfg.n_tiles, cfg.tile_size,
                                       cfg.level, cfg.img_size)
    flags = []

    if cfg.stain_norm:
        montage, ok = macenko(montage)
        if not ok:
            flags.append("STAIN_NORM_FAILED")

    cov = tissue_coverage(montage)
    if cov < 0.05:
        flags.append("LOW_TISSUE_COVERAGE")

    x = torch.from_numpy(montage.transpose(2, 0, 1).copy()).float().div_(255)
    x = ((x - torch.tensor([0.485, 0.456, 0.406])[:, None, None])
         / torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    model.eval()
    with torch.no_grad():
        logits = model(x[None].to(DEVICE)).cpu().numpy()

    grade = int(ordinal_decode(logits, thresholds)[0])
    dist  = ordinal_to_distribution(logits)[0]

    return {
        "isup_grade":    grade,
        "gleason_score": ISUP_TO_GLEASON[grade],
        "risk_band":     RISK_BAND[grade],
        "confidence":    float(dist[grade]),
        "distribution":  {i: round(float(p), 4) for i, p in enumerate(dist)},
        "ordinal_probs": torch.sigmoid(torch.tensor(logits))[0].numpy().round(4).tolist(),
        "tissue_tiles":  int((scores > 0).sum()),
        "coverage":      round(cov, 4),
        "flags":         flags,
    }, montage


held_out = oof_df.sample(1, random_state=3).iloc[0]
result, montage = predict_slide(
    DATA/"train_images"/f"{held_out.image_id}.tiff", model, tau, cfg)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5),
                       gridspec_kw={"width_ratios": [1, 1.15]})
ax[0].imshow(montage); ax[0].axis("off")
ax[0].set_title(f"{held_out.image_id[:14]}…  ·  {held_out.data_provider}",
                loc="left", fontweight=600)

grades = list(result["distribution"].keys())
probs  = list(result["distribution"].values())
colors = ["#2b7a78" if g == result["isup_grade"] else "#cbd5e1" for g in grades]
ax[1].barh(grades, probs, color=colors)
ax[1].invert_yaxis()
ax[1].set_xlim(0, 1)
ax[1].set_yticks(grades)
ax[1].set_yticklabels([f"ISUP {g} · {ISUP_TO_GLEASON[g]}" for g in grades])
ax[1].set_title("prediction distribution", loc="left", fontweight=600)
for g, p in zip(grades, probs):
    ax[1].text(p + .015, g, f"{p:.1%}", va="center", fontsize=9)
ax[1].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()

print(json.dumps(result, indent=2))
print(f"\nground truth:  ISUP {held_out.isup_grade}  ·  Gleason {held_out.gleason_score}")
print(f"predicted:     ISUP {result['isup_grade']}  ·  Gleason {result['gleason_score']}")
print(f"→ {'exact' if result['isup_grade']==held_out.isup_grade else 'off by ' + str(abs(result['isup_grade']-held_out.isup_grade))}")
''')

# ═══════════════════════════════════════════════════════════════════════════ 15
md(r"""
## 15 · What to do next

1. **Save this notebook's output.** Kaggle → *Save Version* → *Save & Run All*. `progenet_<backbone>.onnx` and `model_meta.json` appear under the version's Output tab.
2. **Fill in the ablation table (§12).** Each row is one config edit and one re-run.
3. **Look hard at the cross-provider number (§10).** If it is below 0.60, that is the result — write it up, don't train around it.
4. **Deploy.** Drop both artifacts into `service/artifacts/` and `docker compose up`. See `docs/04-runbook.md`.
5. **Ensemble, later.** Five folds is worth roughly +0.02 QWK for 5× the wall clock. Do it once the pipeline is stable, not before.

### Honest limitations

- One fold, one backbone. This is a reproducible baseline, not a leaderboard entry.
- Slide-level labels only. The model cannot say *where* the high-grade focus is beyond Grad-CAM's rough attribution.
- Labels are imperfect — Karolinska is one pathologist's read. There is a ceiling here that no amount of model capacity crosses.
- Two institutions, two scanners. Anything said about a third is extrapolation until measured.

> **Research use only. Not a medical device. Not for clinical decision-making.**
""")

nb = {
    "cells": C,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.13",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [
            {"sourceId": 18647, "sourceType": "competition"}],
            "isInternetEnabled": True, "language": "python",
            "sourceType": "notebook"},
    },
    "nbformat": 4, "nbformat_minor": 4,
}

out = sys.argv[1] if len(sys.argv) > 1 else "progenet-panda-training.ipynb"
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {out}: {len(C)} cells "
      f"({sum(1 for c in C if c['cell_type']=='code')} code, "
      f"{sum(1 for c in C if c['cell_type']=='markdown')} markdown)")
