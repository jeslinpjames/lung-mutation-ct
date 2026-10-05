#!/usr/bin/env python3
"""
EGFR Multimodal Training v3.7 - standalone script version, for headless/tmux runs on HPC.

Generated from EGFR_Multimodal_Training_v3_7.ipynb - same logic, same cell order, nothing changed
about the modeling. Two things added for running unattended over ssh/tmux:

  1. Everything that gets print()'d also gets written live to a .txt log file (line-buffered, so it's
     there even if the process gets killed mid-run) - default: training_log.txt in the current directory,
     override with: python3 EGFR_Multimodal_Training_v3_7_script.py my_custom_log.txt
     tqdm's live progress bars still print straight to your terminal as normal (not duplicated into the
     log file - that would just fill it with carriage-return spam).
  2. The sanity-check plot saves to egfr_out/sanity_check_crops.png instead of trying to plt.show(),
     since there's no display in a headless script.

Run it (inside tmux, after activating your venv and setting CUDA_VISIBLE_DEVICES):
    python3 -u EGFR_Multimodal_Training_v3_7_script.py
Then Ctrl+B, D to detach. Reattach any time with: tmux attach -t <session name>
Or just watch it from anywhere without attaching: tail -f training_log.txt
"""
import sys, datetime, traceback
import matplotlib
matplotlib.use("Agg")   # headless - no display available, save-to-file instead

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "training_log.txt"

class _Tee:
    """Writes to the real terminal AND the log file at once. Only wraps stdout (plain print()
    statements) - stderr (where tqdm draws its live bars) is left alone so the log file doesn't
    fill up with carriage-return spam; you still see the bars live in your terminal as normal."""
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_log_file = open(LOG_PATH, "a", buffering=1)   # line-buffered: written to disk immediately, not batched
_log_file.write(f"\n\n========== RUN STARTED {datetime.datetime.now().isoformat()} ==========\n")
sys.stdout = _Tee(sys.__stdout__, _log_file)

def _log_uncaught(exc_type, exc_value, exc_tb):
    """Any crash gets written to the log file too, not just lost when the terminal disappears."""
    _log_file.write(f"\n\n========== UNCAUGHT EXCEPTION {datetime.datetime.now().isoformat()} ==========\n")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=_log_file)
    _log_file.flush()
    sys.__excepthook__(exc_type, exc_value, exc_tb)
sys.excepthook = _log_uncaught

print(f"Logging to: {LOG_PATH}  (tail -f {LOG_PATH} from another session to watch progress)")



print(">>>> EGFR 'Virtual Biopsy' — v3.7: same as v3.6, TCGA-LUAD path auto-detection")


print(">>>> 1. Install")

import sys
# !{sys.executable} -m pip install -q "pydicom<3" pydicom-seg SimpleITK scipy scikit-learn scikit-image matplotlib tqdm torchvision requests
print("done")
# !pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

print(">>>> 2. Imports")

import os, glob, shutil, random, warnings, math, io, zipfile, json, numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
import pydicom, pydicom_seg, SimpleITK as sitk
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk as skdisk, ball as skball
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x
warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch", torch.__version__, "| Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "| CUDA:", torch.version.cuda)
else:
    print("WARNING: no CUDA GPU detected - training will be much slower on CPU.")

print(">>>> 3. CONFIG")

DATA_ROOT    = Path("data/nsclc_radiogenomics")
CLINICAL_CSV = Path("data/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
AIM_DIR      = Path("data/AIM_files_updated-11-10-2020/AIM_files_updated-11-10-2020")
TCGA_BASE    = Path("data/tcga_luad"); TCGA_BASE.mkdir(parents=True, exist_ok=True)

def _find_tcga_root(base=TCGA_BASE, max_depth=3):
    """NBIA Data Retriever doesn't always put patient folders directly under the destination you pick -
    it can nest them under an extra subfolder, and stray partial folders from earlier test runs can also
    be lying around. Scan a few levels deep and use whichever folder actually contains the most
    TCGA-XX-XXXX patient directories, instead of assuming a fixed layout."""
    candidates = [base]
    for depth in range(1, max_depth+1):
        candidates += [p for p in base.glob("/".join(["*"]*depth)) if p.is_dir()]
    best, best_n = base, -1
    for c in candidates:
        try:
            n = sum(1 for d in c.iterdir() if d.is_dir() and d.name.upper().startswith("TCGA-"))
        except Exception:
            n = 0
        if n > best_n: best, best_n = c, n
    return best, best_n

TCGA_ROOT, _tcga_n_found = _find_tcga_root()
print(f"TCGA-LUAD patient folder auto-detected: {TCGA_ROOT}  ({_tcga_n_found} patient directories found)")
TCGA_MANUAL_LABELS_CSV = Path("data/tcga_luad_egfr_manual.csv")   # manual fallback, see Section 4
CACHE_BASE   = Path("egfr_cache")
OUTPUT_DIR   = Path("egfr_out"); OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL, POS_LABEL, NEG_LABEL = "EGFR mutation status", "Mutant", "Wildtype"

TARGET_SHAPE   = (32, 64, 64); CROP_PAD_VOX = 16
HU_MIN, HU_MAX = -1000, 400
REQUIRE_SEG    = False
USE_AIM_COORD_FALLBACK = True
USE_FRAME_FALLBACK     = True
ALLOW_CENTER_CROP_FALLBACK = False
AIM_DEPTH_SLICES = TARGET_SHAPE[0]
AIM_LUNG_MARGIN  = 20
FORCE_REBUILD  = False

INCLUDE_HISTOLOGY = False
INCLUDE_ETHNICITY = True

IMAGE_BACKBONE = "2.5d"
PRETRAINED     = True
IMG_FEAT_DIM   = 64; TAB_HIDDEN = 64; FUSION_HIDDEN = 64; DROPOUT = 0.4

N_SLICES        = 9
USE_GATED_FUSION = True
USE_FOCAL_LOSS   = True
FOCAL_GAMMA      = 2.0
USE_DANN         = True
DANN_LAMBDA_MAX  = 0.6
HARMONIZE_RADIOMICS = True

# ---------- NEW in v3.6 ----------
CBIO_BASE  = "https://www.cbioportal.org/api"
CBIO_STUDY = "luad_tcga_pan_can_atlas_2018"     # TCGA-LUAD, PanCancer Atlas
TCIA_BASE  = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TCIA_COLLECTION = "TCGA-LUAD"
MAX_TCGA_PATIENTS = None       # cap for a quick test run, e.g. 40; None = download all matched patients
USE_ROTNET_PRETRAIN = True     # self-supervised pretraining of the image backbone (Section 16)
ROTNET_EPOCHS = 15

N_FOLDS = 5; N_REPEATS = 5
EPOCHS  = 60; LR = 1e-4; WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8; USE_AUGMENTATION = True; USE_CLASS_WEIGHTS = True; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PRE_TAG = f"seg{int(REQUIRE_SEG)}_aimdynbox_{TARGET_SHAPE[0]}x{TARGET_SHAPE[1]}x{TARGET_SHAPE[2]}_pad{CROP_PAD_VOX}_3site"
CACHE_DIR = CACHE_BASE / PRE_TAG
if FORCE_REBUILD and CACHE_DIR.exists(): shutil.rmtree(CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("cache:", CACHE_DIR)

print(">>>> 4. Download TCGA-LUAD EGFR labels (cBioPortal) — NEW")

def fetch_tcga_luad_egfr_labels():
    samples_r = requests.get(f"{CBIO_BASE}/studies/{CBIO_STUDY}/samples",
                              params={"projection":"SUMMARY"}, timeout=60)
    samples_r.raise_for_status()
    samples = pd.DataFrame(samples_r.json())

    body = {"entrezGeneIds":[1956], "sampleListId": f"{CBIO_STUDY}_all"}   # 1956 = EGFR (HUGO Entrez ID)
    mut_r = requests.post(f"{CBIO_BASE}/molecular-profiles/{CBIO_STUDY}_mutations/mutations/fetch",
                           params={"projection":"SUMMARY"}, json=body, timeout=120)
    mut_r.raise_for_status()
    muts = pd.DataFrame(mut_r.json())

    mutated_patients = set(muts["patientId"].unique()) if len(muts) else set()
    out = samples[["patientId"]].drop_duplicates().rename(columns={"patientId":"Case ID"}).copy()
    out["label"] = out["Case ID"].isin(mutated_patients).astype(int)
    return out

def fetch_tcga_luad_clinical():
    r = requests.get(f"{CBIO_BASE}/studies/{CBIO_STUDY}/clinical-data",
                      params={"clinicalDataType":"PATIENT", "projection":"SUMMARY"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty: return pd.DataFrame(columns=["Case ID"])
    piv = df.pivot_table(index="patientId", columns="clinicalAttributeId", values="value",
                          aggfunc="first").reset_index().rename(columns={"patientId":"Case ID"})
    return piv

try:
    tcga_labels = fetch_tcga_luad_egfr_labels()
    tcga_clin_raw = fetch_tcga_luad_clinical()
    print(f"cBioPortal OK: {len(tcga_labels)} samples, {tcga_labels['label'].sum()} EGFR-mutant, "
          f"{tcga_clin_raw.shape[1]-1} clinical attributes")
    SOURCE = "api"
except Exception as e:
    print(f"cBioPortal API call failed ({type(e).__name__}: {e}). "
          f"Looking for manual fallback at {TCGA_MANUAL_LABELS_CSV} ...")
    if TCGA_MANUAL_LABELS_CSV.exists():
        tcga_labels = pd.read_csv(TCGA_MANUAL_LABELS_CSV)[["Case ID","label"]]
        tcga_clin_raw = pd.DataFrame({"Case ID": tcga_labels["Case ID"]})
        print(f"Loaded manual fallback: {len(tcga_labels)} patients")
        SOURCE = "manual"
    else:
        tcga_labels = pd.DataFrame(columns=["Case ID","label"])
        tcga_clin_raw = pd.DataFrame(columns=["Case ID"])
        print("No manual fallback found either - TCGA-LUAD will be skipped. "
              "See the instructions at the top of this notebook to add it.")
        SOURCE = "none"

tcga_labels["Case ID"] = tcga_labels["Case ID"].astype(str).str.strip()
if len(tcga_clin_raw): tcga_clin_raw["Case ID"] = tcga_clin_raw["Case ID"].astype(str).str.strip()

print(">>>> 5. Download TCGA-LUAD CT images (TCIA) — NEW")

def tcia_get(endpoint, params=None):
    r = requests.get(f"{TCIA_BASE}/{endpoint}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()

def list_tcga_luad_ct_series():
    series = tcia_get("getSeries", {"Collection": TCIA_COLLECTION, "Modality": "CT"})
    return pd.DataFrame(series)

def download_series(series_uid, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.dcm")): return True
    r = requests.get(f"{TCIA_BASE}/getImage", params={"SeriesInstanceUID": series_uid}, timeout=900)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out_dir)
    return True

tcga_patients_on_disk = sorted([p.name for p in TCGA_ROOT.iterdir()
                                 if p.is_dir() and p.name.upper().startswith("TCGA-")]) if TCGA_ROOT.exists() else []

if len(tcga_patients_on_disk) >= 5:
    print(f"Found {len(tcga_patients_on_disk)} TCGA-LUAD patients already on disk under {TCGA_ROOT} "
          f"- skipping the TCIA download entirely.")
else:
    tcga_series = pd.DataFrame()
    if len(tcga_labels):
        try:
            tcga_series = list_tcga_luad_ct_series()
            print(f"TCIA OK: {len(tcga_series)} CT series across the {TCIA_COLLECTION} collection")
        except Exception as e:
            print(f"TCIA API call failed ({type(e).__name__}: {e}). "
                  f"Use the manual NBIA Data Retriever path described at the top of this notebook, "
                  f"then just re-run this cell - it will detect files already on disk and skip downloading.")

    if len(tcga_series):
        # one series per patient: prefer the CT series with the most images (usually the diagnostic chest CT)
        tcga_series["PatientID"] = tcga_series["PatientId"] if "PatientId" in tcga_series.columns else tcga_series.get("PatientID")
        best_series = (tcga_series.sort_values("ImageCount", ascending=False)
                                   .drop_duplicates(subset="PatientID", keep="first"))
        wanted = set(tcga_labels["Case ID"])
        best_series = best_series[best_series["PatientID"].isin(wanted)]
        if MAX_TCGA_PATIENTS: best_series = best_series.head(MAX_TCGA_PATIENTS)
        print(f"Downloading {len(best_series)} patients' CT series (this can take a while)...")
        for _, row in tqdm(best_series.iterrows(), total=len(best_series), desc="TCIA download"):
            pid = row["PatientID"]; out_dir = TCGA_ROOT/pid
            try:
                download_series(row["SeriesInstanceUID"], out_dir)
            except Exception as e:
                print(f"  {pid}: download failed ({type(e).__name__})")
    else:
        print("Skipping TCIA download (no series list). Checking for anything already on disk under", TCGA_ROOT)

    tcga_patients_on_disk = sorted([p.name for p in TCGA_ROOT.iterdir()
                                     if p.is_dir() and p.name.upper().startswith("TCGA-")]) if TCGA_ROOT.exists() else []

print(f"TCGA-LUAD patients with CT data on disk: {len(tcga_patients_on_disk)}")

print(">>>> 6. NSCLC-Radiogenomics clinical labels + tabular features (unchanged)")

clin = pd.read_csv(CLINICAL_CSV)
clin["Case ID"] = clin["Case ID"].astype(str).str.strip()
clin["label"] = clin[TARGET_COL].map({POS_LABEL: 1, NEG_LABEL: 0})
clin = clin.dropna(subset=["label"]).copy(); clin["label"] = clin["label"].astype(int)
clin["Patient affiliation"] = clin["Patient affiliation"].astype(str)
print("NSCLC-Radiogenomics usable EGFR labels:", len(clin))
print(clin["label"].value_counts().rename({0:NEG_LABEL,1:POS_LABEL}))

def build_tabular(df):
    f = pd.DataFrame(index=df.index)
    age = pd.to_numeric(df["Age at Histological Diagnosis"], errors="coerce")
    f["age"] = age.fillna(age.median())/100.0
    f["male"] = (df["Gender"].astype(str).str.strip().str.lower()=="male").astype(float)
    sm = df["Smoking status"].astype(str).str.strip()
    for v in ["Nonsmoker","Former","Current"]: f[f"smk_{v}"] = (sm==v).astype(float)
    for c in [c for c in df.columns if c.startswith("Tumor Location")]:
        key = c.split("choice=")[-1].rstrip(")").strip().replace(" ","_")
        f[f"loc_{key}"] = (df[c].astype(str).str.strip().str.lower()=="checked").astype(float)
    f = pd.concat([f, pd.get_dummies(df["%GG"].astype(str).str.strip(), prefix="gg").astype(float)], axis=1)
    if INCLUDE_ETHNICITY and "Ethnicity" in df.columns:
        f = pd.concat([f, pd.get_dummies(df["Ethnicity"].astype(str).str.strip(), prefix="eth").astype(float)], axis=1)
    if INCLUDE_HISTOLOGY:
        f = pd.concat([f, pd.get_dummies(df["Histology "].astype(str).str.strip(), prefix="hist").astype(float)], axis=1)
    f["clin_detail_available"] = 1.0
    return f
print("NSCLC-Radiogenomics tabular features:", build_tabular(clin).shape[1])

print(">>>> 7. TCGA-LUAD tabular features — NEW")

def build_tabular_tcga(df):
    f = pd.DataFrame(index=df.index)
    age_col = next((c for c in df.columns if c.upper() in ("AGE","AGE_AT_DIAGNOSIS")), None)
    sex_col = next((c for c in df.columns if c.upper() in ("SEX","GENDER")), None)
    age = pd.to_numeric(df[age_col], errors="coerce") if age_col else pd.Series(np.nan, index=df.index)
    f["age"] = age.fillna(age.median() if age.notna().any() else 65)/100.0
    if sex_col:
        f["male"] = (df[sex_col].astype(str).str.strip().str.upper()=="MALE").astype(float)
    else:
        f["male"] = 0.5   # unknown -> neutral value rather than a false default
    f["clin_detail_available"] = 0.0
    return f
tcga_tab_preview = build_tabular_tcga(tcga_clin_raw) if len(tcga_clin_raw) else pd.DataFrame()
print("TCGA-LUAD tabular preview:", tcga_tab_preview.shape if len(tcga_tab_preview) else "no data")

print(">>>> 8. Index NSCLC-Radiogenomics DICOM + match AIM files (unchanged)")

ALL_AIM = glob.glob(os.path.join(str(AIM_DIR), "*.xml")) if AIM_DIR.exists() else []
def aim_file_for(cid):
    mf = [f for f in ALL_AIM if cid in os.path.basename(f)]
    return mf[0] if mf else None

def scan_patient(pdir):
    info = {"ct_dir": None, "ct_n": 0, "seg_file": None, "seg_modality": None}
    for root, _, files in os.walk(pdir):
        dcms = [f for f in files if f.lower().endswith(".dcm")]
        if not dcms: continue
        try: ds = pydicom.dcmread(os.path.join(root, dcms[0]), stop_before_pixels=True, force=True)
        except Exception: continue
        mod = str(getattr(ds, "Modality", "")).upper()
        if mod == "CT":
            if len(dcms) > info["ct_n"]: info["ct_dir"], info["ct_n"] = root, len(dcms)
        elif mod in ("SEG","RTSTRUCT"):
            info["seg_file"], info["seg_modality"] = os.path.join(root, dcms[0]), mod
    return info

records = []
for cid in tqdm(clin["Case ID"].tolist(), desc="indexing NSCLC-R01"):
    pdir = DATA_ROOT/cid
    info = scan_patient(pdir) if pdir.exists() else {"ct_dir":None,"ct_n":0,"seg_file":None,"seg_modality":None}
    info["Case ID"] = cid; info["aim_file"] = aim_file_for(cid); records.append(info)
nsclc_cohort = clin.merge(pd.DataFrame(records), on="Case ID", how="left")
print("CT:", nsclc_cohort["ct_dir"].notna().sum(), "| SEG:", (nsclc_cohort["seg_modality"]=="SEG").sum(),
      "| AIM:", nsclc_cohort["aim_file"].notna().sum())

print(">>>> 9. AIM recovery — global SOP search + lung-gated dynamic box (unchanged)")

def _local(t): return t.rsplit('}',1)[-1]

def parse_aim_markup(xml_path):
    out=[]
    try: root=ET.parse(xml_path).getroot()
    except Exception: return out
    for me in root.iter():
        if _local(me.tag)!="MarkupEntity": continue
        sop=None;frame=None;pts=[]
        for el in me.iter():
            lt=_local(el.tag)
            if lt=="imageReferenceUid": sop=el.get("root")
            elif lt=="referencedFrameNumber": frame=el.get("value")
            elif lt=="TwoDimensionSpatialCoordinate":
                x=y=None
                for c in el:
                    if _local(c.tag)=="x": x=float(c.get("value"))
                    if _local(c.tag)=="y": y=float(c.get("value"))
                if x is not None and y is not None: pts.append((x,y))
        if pts: out.append({"sop":sop,"frame":int(frame) if frame else None,"xy":pts})
    return out

def find_series_for_sops(pdir, target_sops):
    found={}
    for root,_,files in os.walk(pdir):
        if len(found)==len(target_sops): break
        dcms=[f for f in files if f.lower().endswith(".dcm")]
        if not dcms: continue
        try:
            t=pydicom.dcmread(os.path.join(root,dcms[0]),stop_before_pixels=True,force=True)
            if str(getattr(t,"Modality","")).upper()!="CT": continue
        except Exception: continue
        for dcm in dcms:
            try:
                ds=pydicom.dcmread(os.path.join(root,dcm),stop_before_pixels=True,force=True)
                uid=str(getattr(ds,'SOPInstanceUID',''))
                if uid in target_sops: found[uid]=root
            except Exception: pass
    return found

def _lung_gate_2d(sl_hu, body_thresh=-500, air_thresh=-400, margin=AIM_LUNG_MARGIN):
    bdy=ndimage.binary_fill_holes(sl_hu>body_thresh)
    lbl,n=ndimage.label(bdy)
    if n==0: return np.zeros(sl_hu.shape,bool)
    bdy=lbl==(np.argmax(ndimage.sum(np.ones_like(lbl),lbl,range(1,n+1)))+1)
    air=ndimage.binary_opening((sl_hu<air_thresh)&bdy,structure=skdisk(2))
    lbl,n=ndimage.label(air)
    if n==0: return np.zeros(sl_hu.shape,bool)
    sizes=ndimage.sum(np.ones_like(lbl),lbl,range(1,n+1))
    keep=np.where(sizes>=64)[0]+1
    lung=np.isin(lbl,keep) if keep.size else (lbl==np.argmax(sizes)+1)
    return ndimage.binary_dilation(lung,structure=skdisk(margin))

def estimate_tumor_bbox(sl_hu, cx, cy, r_max=55, hu_thresh=-400, min_half=12, pad=10,
                        fallback_half=30, lung_gate=None):
    H,W=sl_hu.shape; cyi,cxi=int(round(cy)),int(round(cx))
    y0,y1=max(0,cyi-r_max),min(H,cyi+r_max+1); x0,x1=max(0,cxi-r_max),min(W,cxi+r_max+1)
    win=sl_hu[y0:y1,x0:x1]; ly,lx=cyi-y0,cxi-x0
    if lung_gate is None: lung_gate=_lung_gate_2d(sl_hu)
    if lung_gate is not None and lung_gate.any():
        gw=lung_gate[y0:y1,x0:x1]
        if not gw[ly,lx]: return max(min_half,fallback_half)
    else: gw=None
    m=win>hu_thresh
    yy,xx=np.ogrid[:win.shape[0],:win.shape[1]]
    disk_win=(yy-ly)**2+(xx-lx)**2<=r_max**2; m&=disk_win
    if gw is not None: m&=gw
    m=ndimage.binary_opening(m,structure=skdisk(2))
    comp=None
    if not m[ly,lx]:
        md_=ndimage.binary_dilation(m,structure=skdisk(2))
        if md_[ly,lx]: m=md_
    if m[ly,lx]:
        lbl,_=ndimage.label(m); comp=lbl==lbl[ly,lx]
        ys,xs=np.where(comp); h=ys.max()-ys.min()+1; w=xs.max()-xs.min()+1
        fill=comp.sum()/float(h*w); aspect=max(h,w)/float(max(1,min(h,w))); raw_half=int(np.ceil(max(h,w)/2))+pad
        if (comp.sum()>0.55*disk_win.sum()) or fill<0.35 or aspect>3.0 or raw_half>=r_max: comp=None
    if comp is None or comp.sum()<9: return max(min_half,fallback_half)
    ys,xs=np.where(comp)
    half=int(np.ceil(max(ys.max()-ys.min(),xs.max()-xs.min())/2))+pad
    return int(np.clip(half,min_half,r_max-1))

def _finish_cube(crop):
    if crop.size==0 or min(crop.shape)<2: return None
    crop=ndimage.zoom(crop.astype(np.float32),[t/max(1,s) for t,s in zip(TARGET_SHAPE,crop.shape)],order=1)
    crop=np.clip(crop,HU_MIN,HU_MAX)
    return ((crop-HU_MIN)/(HU_MAX-HU_MIN)).astype(np.float32)

def _aim_cube_from_series(series_dir, mk, use_frame=False):
    r=sitk.ImageSeriesReader(); files=r.GetGDCMSeriesFileNames(series_dir)
    if not files: return None
    z_idx=None; cols=rows=None
    if not use_frame:
        for z,fp in enumerate(files):
            ds=pydicom.dcmread(fp,stop_before_pixels=True,force=True)
            if str(getattr(ds,'SOPInstanceUID',''))==mk["sop"]:
                z_idx=z; cols=getattr(ds,'Columns',None); rows=getattr(ds,'Rows',None); break
    if z_idx is None:
        if use_frame and mk["frame"] is not None:
            z_idx=int(np.clip(mk["frame"]-1,0,len(files)-1))
            ds=pydicom.dcmread(files[0],stop_before_pixels=True,force=True)
            cols=getattr(ds,'Columns',None); rows=getattr(ds,'Rows',None)
        else: return None
    r.SetFileNames(files); arr=sitk.GetArrayFromImage(r.Execute()); Z,Y,X=arr.shape
    sx=X/float(cols) if cols else 1.0; sy=Y/float(rows) if rows else 1.0
    cx,cy=mk["xy"][0][0]*sx, mk["xy"][0][1]*sy
    half=estimate_tumor_bbox(arr[z_idx], cx, cy)
    dz=AIM_DEPTH_SLICES//2
    y0,y1=max(0,int(cy)-half),min(Y,int(cy)+half); x0,x1=max(0,int(cx)-half),min(X,int(cx)+half)
    z0,z1=max(0,z_idx-dz),min(Z,z_idx+dz)
    return _finish_cube(arr[z0:z1,y0:y1,x0:x1])

def recover_aim_cube(pdir, markups, ct_dir_default):
    target_sops={mk["sop"] for mk in markups if mk["sop"]}
    sop_dir=find_series_for_sops(pdir, target_sops) if target_sops else {}
    for mk in markups:
        if mk["sop"] in sop_dir:
            cube=_aim_cube_from_series(sop_dir[mk["sop"]], mk)
            if cube is not None: return cube,"ok_aim"
    if USE_FRAME_FALLBACK and isinstance(ct_dir_default,str):
        for mk in markups:
            if mk["sop"] not in sop_dir and mk["frame"] is not None:
                cube=_aim_cube_from_series(ct_dir_default, mk, use_frame=True)
                if cube is not None: return cube,"ok_aim_frame"
    return None,"aim_unresolved"

print(">>>> 10. Preprocess NSCLC-Radiogenomics -> tumour cubes (SEG first, AIM fallback — unchanged)")

def load_ct(ct_dir):
    r=sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(str(ct_dir))); return r.Execute()
def seg_mask_on_ct(seg_file, ct_img):
    result=pydicom_seg.SegmentReader().read(pydicom.dcmread(seg_file)); mask=None
    for num in result.available_segments:
        res=sitk.Resample(result.segment_image(num),ct_img,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        a=sitk.GetArrayFromImage(res).astype(bool); mask=a if mask is None else (mask|a)
    return mask
def crop_to_cube(ct_arr, mask):
    zz,yy,xx=np.where(mask); p=CROP_PAD_VOX
    z0,z1=max(0,zz.min()-p),min(ct_arr.shape[0],zz.max()+p+1)
    y0,y1=max(0,yy.min()-p),min(ct_arr.shape[1],yy.max()+p+1)
    x0,x1=max(0,xx.min()-p),min(ct_arr.shape[2],xx.max()+p+1)
    return _finish_cube(ct_arr[z0:z1,y0:y1,x0:x1])

status=[]
for _,row in tqdm(nsclc_cohort.iterrows(), total=len(nsclc_cohort), desc="preprocess NSCLC-R01"):
    cid=row["Case ID"]; out=CACHE_DIR/f"{cid}.npy"
    if out.exists(): status.append((cid,"cached")); continue
    if pd.isna(row["ct_dir"]): status.append((cid,"no_ct")); continue
    try:
        ct=load_ct(row["ct_dir"]); ct_arr=sitk.GetArrayFromImage(ct); mask=None
        if row["seg_modality"]=="SEG":
            try: mask=seg_mask_on_ct(row["seg_file"],ct)
            except Exception: mask=None
        if mask is not None and mask.any():
            cube=crop_to_cube(ct_arr,mask)
            if cube is not None: np.save(out,cube); status.append((cid,"ok_seg")); continue
        if USE_AIM_COORD_FALLBACK and isinstance(row["aim_file"],str):
            markups=parse_aim_markup(row["aim_file"])
            if markups:
                cube,stat=recover_aim_cube(str(DATA_ROOT/cid), markups, row["ct_dir"])
                if cube is not None: np.save(out,cube); status.append((cid,stat)); continue
                status.append((cid,"aim_unresolved")); continue
            status.append((cid,"aim_no_markup")); continue
        status.append((cid,"no_seg_skipped"))
    except Exception as e: status.append((cid,f"error:{type(e).__name__}"))

st=pd.DataFrame(status,columns=["Case ID","status"]); print(st["status"].value_counts())
nsclc_cohort=nsclc_cohort.merge(st,on="Case ID",how="left")
nsclc_cohort=nsclc_cohort[nsclc_cohort["status"].isin(["ok_seg","ok_aim","ok_aim_frame","cached"])].reset_index(drop=True)
print("\nNSCLC-Radiogenomics final:",len(nsclc_cohort))
print(nsclc_cohort.groupby("Patient affiliation")["label"].agg(["size","sum"]))

print(">>>> 11. Preprocess TCGA-LUAD -> tumour cubes — NEW: heuristic auto-localization")

def _lung_mask_3d(ct_arr, body_thresh=-500, air_thresh=-400):
    body = ndimage.binary_fill_holes(ct_arr > body_thresh)
    lbl, n = ndimage.label(body)
    if n == 0: return np.zeros_like(ct_arr, bool)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n+1))
    body = lbl == (int(np.argmax(sizes)) + 1)
    air = (ct_arr < air_thresh) & body
    air = ndimage.binary_opening(air, structure=np.ones((1,3,3)))
    lbl2, n2 = ndimage.label(air)
    if n2 == 0: return np.zeros_like(ct_arr, bool)
    sizes2 = ndimage.sum(np.ones_like(lbl2), lbl2, range(1, n2+1))
    keep = np.where(sizes2 >= 500)[0] + 1
    lung = np.isin(lbl2, keep) if keep.size else (lbl2 == int(np.argmax(sizes2))+1)
    return ndimage.binary_closing(lung, structure=np.ones((3,5,5)))

def auto_localize_tumor_cube(ct_arr, min_diam_vox=8, max_diam_vox=65):
    """Unsupervised tumour-candidate localizer for cohorts with no expert annotation. See markdown above."""
    lung = _lung_mask_3d(ct_arr)
    if not lung.any(): return None
    tissue = (ct_arr > -150) & (ct_arr < 150) & lung
    tissue = ndimage.binary_opening(tissue, structure=skball(1))
    lbl, n = ndimage.label(tissue)
    if n == 0: return None
    best=None; best_score=-1
    for i in range(1, n+1):
        comp = lbl==i; vol_vox = comp.sum()
        if vol_vox < 27: continue
        zz,yy,xx = np.where(comp)
        bb = np.array([zz.max()-zz.min()+1, yy.max()-yy.min()+1, xx.max()-xx.min()+1])
        diam = bb.max()
        if diam < min_diam_vox or diam > max_diam_vox: continue
        aspect = bb.max()/max(1,bb.min())
        if aspect > 2.5: continue
        fill = vol_vox/float(np.prod(bb))
        if fill < 0.3: continue
        if vol_vox > best_score: best_score=vol_vox; best=(zz,yy,xx)
    return best

def crop_from_coords(ct_arr, coords):
    zz,yy,xx = coords; p=CROP_PAD_VOX
    z0,z1=max(0,zz.min()-p),min(ct_arr.shape[0],zz.max()+p+1)
    y0,y1=max(0,yy.min()-p),min(ct_arr.shape[1],yy.max()+p+1)
    x0,x1=max(0,xx.min()-p),min(ct_arr.shape[2],xx.max()+p+1)
    return _finish_cube(ct_arr[z0:z1,y0:y1,x0:x1])

tcga_cohort = tcga_labels.merge(tcga_clin_raw, on="Case ID", how="left") if len(tcga_labels) else pd.DataFrame(columns=["Case ID","label"])
tcga_records=[]
for pid in tqdm(tcga_patients_on_disk, desc="indexing TCGA-LUAD"):
    pdir = TCGA_ROOT/pid
    info = scan_patient(pdir)
    info["Case ID"]=pid; tcga_records.append(info)
if tcga_records:
    tcga_cohort = tcga_cohort.merge(pd.DataFrame(tcga_records), on="Case ID", how="inner")
else:
    tcga_cohort = tcga_cohort.iloc[0:0]

tcga_status=[]
for _,row in tqdm(tcga_cohort.iterrows(), total=len(tcga_cohort), desc="preprocess TCGA-LUAD"):
    cid=row["Case ID"]; out=CACHE_DIR/f"{cid}.npy"
    if out.exists(): tcga_status.append((cid,"cached")); continue
    if pd.isna(row.get("ct_dir")): tcga_status.append((cid,"no_ct")); continue
    try:
        ct=load_ct(row["ct_dir"]); ct_arr=sitk.GetArrayFromImage(ct)
        coords=auto_localize_tumor_cube(ct_arr)
        if coords is None: tcga_status.append((cid,"auto_localize_failed")); continue
        cube=crop_from_coords(ct_arr, coords)
        if cube is None: tcga_status.append((cid,"crop_failed")); continue
        np.save(out,cube); tcga_status.append((cid,"ok_auto"))
    except Exception as e: tcga_status.append((cid,f"error:{type(e).__name__}"))

if tcga_status:
    tst=pd.DataFrame(tcga_status,columns=["Case ID","status"]); print(tst["status"].value_counts())
    tcga_cohort=tcga_cohort.merge(tst,on="Case ID",how="left")
    tcga_cohort=tcga_cohort[tcga_cohort["status"].isin(["ok_auto","cached"])].reset_index(drop=True)
    tcga_cohort["Patient affiliation"]="TCGA-LUAD"
print("\nTCGA-LUAD final:", len(tcga_cohort))
if len(tcga_cohort): print(tcga_cohort["label"].value_counts().rename({0:NEG_LABEL,1:POS_LABEL}))

print(">>>> 12. Combine all three sites into one cohort — NEW")

nsclc_tab = build_tabular(nsclc_cohort)
tcga_tab  = build_tabular_tcga(tcga_cohort) if len(tcga_cohort) else pd.DataFrame()

all_cols = sorted(set(nsclc_tab.columns) | set(tcga_tab.columns))
nsclc_tab = nsclc_tab.reindex(columns=all_cols, fill_value=0.0)
tcga_tab  = tcga_tab.reindex(columns=all_cols, fill_value=0.0) if len(tcga_tab) else pd.DataFrame(columns=all_cols)

nsclc_keep = ["Case ID","label","Patient affiliation"]
tcga_keep  = ["Case ID","label","Patient affiliation"]
cohort = pd.concat([
    nsclc_cohort[nsclc_keep].reset_index(drop=True).join(nsclc_tab.reset_index(drop=True)),
    tcga_cohort[tcga_keep].reset_index(drop=True).join(tcga_tab.reset_index(drop=True)) if len(tcga_cohort) else pd.DataFrame(columns=nsclc_keep+all_cols)
], ignore_index=True)

TAB_COLS = all_cols
print("Combined cohort:", len(cohort))
print(cohort.groupby("Patient affiliation")["label"].agg(["size","sum"]))
print("Tabular columns:", len(TAB_COLS))

print(">>>> 13. SANITY — eyeball crops from all three sources")

import matplotlib.pyplot as plt
aim_ids=nsclc_cohort.loc[nsclc_cohort["status"].isin(["ok_aim","ok_aim_frame"]),"Case ID"].tolist()[:3]
seg_ids=nsclc_cohort.loc[nsclc_cohort["status"].isin(["ok_seg","cached"]),"Case ID"].tolist()[:3]
tcga_ids=tcga_cohort["Case ID"].tolist()[:4] if len(tcga_cohort) else []
show=[("SEG",c) for c in seg_ids]+[("AIM",c) for c in aim_ids]+[("TCGA-auto",c) for c in tcga_ids]
if show:
    n=len(show); fig,ax=plt.subplots(1,n,figsize=(2.2*n,2.4))
    if n==1: ax=[ax]
    for a,(src,cid) in zip(ax,show):
        cube=np.load(CACHE_DIR/f"{cid}.npy"); a.imshow(cube[cube.shape[0]//2],cmap="gray")
        a.set_title(f"{src}\n{cid}",fontsize=7); a.axis("off")
    plt.tight_layout()
    sanity_png = OUTPUT_DIR/"sanity_check_crops.png"
    plt.savefig(str(sanity_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sanity-check crop images to {sanity_png}")

print(">>>> 14. Radiomics features + per-site harmonization (unchanged logic, now spans 3 sites)")

def tumor_roi(cube, thr=0.5):
    m=cube>thr; D,H,W=cube.shape
    if not m.any():
        m=np.zeros_like(cube,bool); m[D//4:3*D//4,H//4:3*H//4,W//4:3*W//4]=True; return m
    lbl,n=ndimage.label(m); cl=lbl[D//2,H//2,W//2]
    if cl>0: return lbl==cl
    sizes=ndimage.sum(np.ones_like(lbl),lbl,index=range(1,n+1))
    return lbl==(1+int(np.argmax(sizes)))

def radiomics_features(cube):
    f={}; roi=tumor_roi(cube); vals=cube[roi]
    if vals.size<8: vals=cube.ravel()
    f["fo_mean"]=vals.mean(); f["fo_std"]=vals.std(); f["fo_median"]=np.median(vals)
    f["fo_p10"]=np.percentile(vals,10); f["fo_p90"]=np.percentile(vals,90)
    f["fo_iqr"]=np.percentile(vals,75)-np.percentile(vals,25)
    f["fo_min"]=vals.min(); f["fo_max"]=vals.max(); f["fo_range"]=vals.max()-vals.min()
    f["fo_mad"]=np.mean(np.abs(vals-vals.mean())); f["fo_skew"]=skew(vals); f["fo_kurt"]=kurtosis(vals)
    f["fo_energy"]=np.mean(vals**2)
    cnt,_=np.histogram(vals,bins=32,range=(0,1)); p=cnt/max(1,cnt.sum()); pp=p[p>0]
    f["fo_entropy"]=-(pp*np.log2(pp)).sum(); f["fo_uniformity"]=(p**2).sum()
    vol=float(roi.sum()); f["sh_volume"]=vol; coords=np.argwhere(roi)
    if len(coords)>0:
        bb=coords.max(0)-coords.min(0)+1
        f["sh_extent"]=vol/float(np.prod(bb)); f["sh_elongation"]=float(bb.min()/max(1,bb.max()))
    else: f["sh_extent"]=0.0; f["sh_elongation"]=0.0
    er=ndimage.binary_erosion(roi); surf=float((roi&~er).sum())
    f["sh_surface"]=surf; f["sh_sa_vol"]=surf/max(1.0,vol)
    f["sh_sphericity"]=float((4*np.pi*((3*vol/(4*np.pi))**(2/3)))/max(1.0,surf)) if vol>0 else 0.0
    areas=roi.sum((1,2)); z=int(np.argmax(areas)) if areas.max()>0 else cube.shape[0]//2
    sl=cube[z]
    if roi[z].any():
        ys,xs=np.where(roi[z]); sl=sl[ys.min():ys.max()+1, xs.min():xs.max()+1]
    levels=16; q=np.clip((sl*(levels-1)).round(),0,levels-1).astype(np.uint8)
    props=["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]
    if min(q.shape)>=3:
        glcm=graycomatrix(q,distances=[1],angles=[0,np.pi/4,np.pi/2,3*np.pi/4],levels=levels,symmetric=True,normed=True)
        for pr in props: f[f"tex_{pr}"]=float(np.nan_to_num(graycoprops(glcm,pr).mean()))
    else:
        for pr in props: f[f"tex_{pr}"]=0.0
    return f

RAD_CACHE=OUTPUT_DIR/f"radiomics_{PRE_TAG}.csv"
if RAD_CACHE.exists():
    rad_df=pd.read_csv(RAD_CACHE).set_index("Case ID")
else:
    rows={}
    for cid in tqdm(cohort["Case ID"], desc="radiomics"):
        rows[cid]=radiomics_features(np.load(CACHE_DIR/f"{cid}.npy"))
    rad_df=pd.DataFrame.from_dict(rows,orient="index"); rad_df.index.name="Case ID"
    rad_df.reset_index().to_csv(RAD_CACHE,index=False)
rad_df=rad_df.reindex(cohort["Case ID"]).fillna(0.0)
rad_mat_raw=np.nan_to_num(rad_df.values.astype(np.float32)); N_RAD=rad_mat_raw.shape[1]
print("Radiomics features:",N_RAD)

SITE_ARR = cohort["Patient affiliation"].astype(str).values
SITE_CODES, SITE_NAMES = pd.factorize(SITE_ARR)

def harmonize_radiomics(raw, train_idx):
    out = raw.copy()
    global_sc = StandardScaler().fit(raw[train_idx])
    for s in np.unique(SITE_CODES):
        site_train = train_idx[SITE_CODES[train_idx]==s]
        site_all   = np.where(SITE_CODES==s)[0]
        if len(site_train) >= 2:
            sc = StandardScaler().fit(raw[site_train]); out[site_all] = sc.transform(raw[site_all])
        else:
            out[site_all] = global_sc.transform(raw[site_all])
    return np.nan_to_num(out).astype(np.float32)

RAD_MAT = harmonize_radiomics(rad_mat_raw, np.arange(len(cohort))) if HARMONIZE_RADIOMICS else \
          np.nan_to_num(StandardScaler().fit_transform(rad_mat_raw)).astype(np.float32)

print(">>>> 15. Tabular matrix + Dataset (N_SITES now 3)")

tab_mat=cohort[TAB_COLS].values.astype(np.float32)
N_TAB=tab_mat.shape[1]; labels=cohort["label"].values.astype(np.float32)
N_SITES = len(SITE_NAMES)
print("Tabular dim:",N_TAB,"| Radiomics dim:",N_RAD,"| N:",len(cohort),"| positives:",int(labels.sum()),
      "| sites:",list(SITE_NAMES))

def augment_cube(cube):
    if random.random()<0.5: cube=torch.flip(cube,dims=[2])
    if random.random()<0.5: cube=torch.flip(cube,dims=[3])
    cube=cube*random.uniform(0.9,1.1)+random.uniform(-0.05,0.05)
    return cube.clamp(0,1)

class EGFRDataset(Dataset):
    def __init__(self, indices, train=False):
        self.indices=list(indices); self.train=train
    def __len__(self): return len(self.indices)
    def __getitem__(self,k):
        i=self.indices[k]
        cube=np.load(CACHE_DIR/f"{cohort.iloc[i]['Case ID']}.npy")
        img=torch.from_numpy(cube).float().unsqueeze(0)
        if self.train and USE_AUGMENTATION: img=augment_cube(img)
        tab=torch.from_numpy(tab_mat[i]).float(); rad=torch.from_numpy(RAD_MAT[i]).float()
        y=torch.tensor([labels[i]],dtype=torch.float32)
        site=torch.tensor(SITE_CODES[i],dtype=torch.long)
        return img,tab,rad,y,site

print(">>>> 16. Model — multi-slice image encoder, gated fusion, GRL + domain head (n_sites=3 now), focal loss")

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__(); self.gamma=gamma; self.pos_weight=pos_weight
    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        pt = torch.where(y==1, p, 1-p)
        w = torch.where(y==1, self.pos_weight, torch.ones_like(p)) if self.pos_weight is not None else 1.0
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (w * ((1-pt)**self.gamma) * bce).mean()

class Image25DEncoder(nn.Module):
    def __init__(self,out_dim=IMG_FEAT_DIM,pretrained=PRETRAINED,p=DROPOUT,n_slices=N_SLICES):
        super().__init__()
        net=tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        self.features=nn.Sequential(*list(net.children())[:-1])
        self.proj=nn.Sequential(nn.Flatten(),nn.Dropout(p),nn.Linear(512,out_dim),nn.ReLU(inplace=True))
        self.n_slices=n_slices
        self.register_buffer("mean",torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self,cube):
        B,_,D,H,W=cube.shape
        idx=np.linspace(0,D-1,self.n_slices).round().astype(int)
        x=cube[:,0][:,idx,:,:]
        x=x.reshape(B*self.n_slices,1,H,W).repeat(1,3,1,1)
        x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False)
        x=(x-self.mean)/self.std
        feat=self.features(x).flatten(1)
        feat=feat.view(B,self.n_slices,-1).mean(1)
        return self.proj(feat)

class Image3DEncoder(nn.Module):
    def __init__(self,out_dim=IMG_FEAT_DIM,p=DROPOUT):
        super().__init__()
        def blk(ci,co): return nn.Sequential(nn.Conv3d(ci,co,3,padding=1),nn.BatchNorm3d(co),nn.ReLU(inplace=True),nn.MaxPool3d(2))
        self.net=nn.Sequential(blk(1,16),blk(16,32),blk(32,64),nn.AdaptiveAvgPool3d(1),nn.Flatten())
        self.proj=nn.Sequential(nn.Dropout(p),nn.Linear(64,out_dim),nn.ReLU(inplace=True))
    def forward(self,x): return self.proj(self.net(x))

ROTNET_BACKBONE_PATH = OUTPUT_DIR/"rotnet_backbone.pt"

def make_image_encoder():
    enc = Image3DEncoder() if IMAGE_BACKBONE=="3d" else Image25DEncoder()
    if IMAGE_BACKBONE!="3d" and USE_ROTNET_PRETRAIN and ROTNET_BACKBONE_PATH.exists():
        sd = torch.load(ROTNET_BACKBONE_PATH, map_location="cpu")
        enc.features.load_state_dict(sd)
    return enc

class MLPEncoder(nn.Module):
    def __init__(self,n_in,hidden=TAB_HIDDEN,out_dim=32,p=DROPOUT):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_in,hidden),nn.ReLU(inplace=True),nn.Dropout(p),
                               nn.Linear(hidden,out_dim),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class GatedFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims=dims; total=sum(dims)
        self.gate=nn.Sequential(nn.Linear(total,len(dims)), nn.Sigmoid())
    def forward(self, feats):
        cat=torch.cat(feats,1)
        g=self.gate(cat)
        gated=[f*g[:,i:i+1] for i,f in enumerate(feats)]
        return torch.cat(gated,1)

class FusionNet(nn.Module):
    def __init__(self,n_tab,n_rad,use_image=True,use_tabular=True,use_radiomics=False,
                 gated=USE_GATED_FUSION, dann=USE_DANN, n_sites=3):
        super().__init__()
        self.use_image,self.use_tabular,self.use_radiomics=use_image,use_tabular,use_radiomics
        self.gated=gated; self.dann=dann and use_image
        dims=[]
        if use_image: self.img=make_image_encoder(); dims.append(IMG_FEAT_DIM)
        if use_tabular: self.tab=MLPEncoder(n_tab); dims.append(32)
        if use_radiomics: self.rad=MLPEncoder(n_rad); dims.append(32)
        dim=sum(dims)
        self.fuse = GatedFusion(dims) if (gated and len(dims)>1) else None
        self.head=nn.Sequential(nn.Linear(dim,FUSION_HIDDEN),nn.ReLU(inplace=True),
                                nn.Dropout(DROPOUT),nn.Linear(FUSION_HIDDEN,1))
        if self.dann:
            self.domain_head=nn.Sequential(nn.Linear(IMG_FEAT_DIM,32),nn.ReLU(inplace=True),
                                            nn.Linear(32,n_sites))
    def forward(self,img,tab,rad,lambd=0.0,return_domain=False):
        f=[]; img_feat=None
        if self.use_image:
            img_feat=self.img(img); f.append(img_feat)
        if self.use_tabular: f.append(self.tab(tab))
        if self.use_radiomics: f.append(self.rad(rad))
        fused = self.fuse(f) if self.fuse is not None else torch.cat(f,1)
        logit = self.head(fused)
        if return_domain and self.dann:
            dom_logit = self.domain_head(grad_reverse(img_feat, lambd))
            return logit, dom_logit
        return logit

print(">>>> 17. Self-supervised RotNet pretraining — NEW")

class RotHead(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone=backbone; self.fc=nn.Linear(512,4)
    def forward(self,x): return self.fc(self.backbone(x).flatten(1))

def make_rotation_batch(cubes):
    B=cubes.shape[0]; D=cubes.shape[2]
    mid=cubes[:,0,D//2,:,:]
    imgs=[]; labs=[]
    for i in range(B):
        k=random.randint(0,3)
        imgs.append(torch.rot90(mid[i],k,dims=[0,1])); labs.append(k)
    imgs=torch.stack(imgs).unsqueeze(1).repeat(1,3,1,1)
    return imgs, torch.tensor(labs,dtype=torch.long)

def pretrain_rotnet(all_indices, epochs=ROTNET_EPOCHS, lr=1e-4, batch_size=16):
    net=tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if PRETRAINED else None)
    backbone=nn.Sequential(*list(net.children())[:-1])
    model=RotHead(backbone).to(DEVICE)
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
    std =torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    loader=DataLoader(EGFRDataset(all_indices,train=False),batch_size=batch_size,shuffle=True,pin_memory=PIN)
    model.train()
    for ep in tqdm(range(epochs),desc="RotNet pretrain"):
        tot=0.0; n=0
        for img,tab,rad,y,site in loader:
            x,rot_y=make_rotation_batch(img)
            x=F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False).to(DEVICE)
            x=(x-mean)/std; rot_y=rot_y.to(DEVICE)
            opt.zero_grad(); logits=model(x); loss=F.cross_entropy(logits,rot_y)
            loss.backward(); opt.step(); tot+=loss.item(); n+=1
        print(f"  RotNet epoch {ep+1}/{epochs} loss {tot/max(1,n):.3f}")
    torch.save(model.backbone.state_dict(), ROTNET_BACKBONE_PATH)
    print("saved backbone ->", ROTNET_BACKBONE_PATH)
    return model.backbone

PIN = (DEVICE=="cuda")
if USE_ROTNET_PRETRAIN:
    pretrain_rotnet(np.arange(len(cohort)))
else:
    print("USE_ROTNET_PRETRAIN is False - skipping, image encoders will use plain ImageNet init.")

print(">>>> 18. Train + predict — single run, no multi-seed")

from torch.cuda.amp import autocast, GradScaler

@torch.no_grad()
def predict(model, loader):
    model.eval(); P,Y=[],[]
    for img,tab,rad,y,site in loader:
        img=img.to(DEVICE,non_blocking=True); tab=tab.to(DEVICE,non_blocking=True); rad=rad.to(DEVICE,non_blocking=True)
        with autocast(enabled=(DEVICE=="cuda")): logit=model(img,tab,rad)
        P.append(torch.sigmoid(logit).float().cpu().numpy().ravel()); Y.append(y.numpy().ravel())
    return np.concatenate(P), np.concatenate(Y)

def dann_lambda(epoch, total_epochs, lambda_max):
    p = epoch/float(max(1,total_epochs))
    return lambda_max * (2.0/(1.0+math.exp(-10*p)) - 1.0)

def train_one(train_idx,val_idx,use_image=True,use_tabular=True,use_radiomics=False,
              target_idx=None, use_dann=None, show_progress=True, desc="train"):
    use_dann = USE_DANN if use_dann is None else use_dann
    use_dann = use_dann and use_image and (target_idx is not None) and (len(target_idx)>0)
    tr=DataLoader(EGFRDataset(train_idx,train=True),batch_size=BATCH_SIZE,shuffle=True,pin_memory=PIN)
    va=DataLoader(EGFRDataset(val_idx,train=False),batch_size=BATCH_SIZE,shuffle=False,pin_memory=PIN)
    if use_dann:
        tgt_loader = DataLoader(EGFRDataset(target_idx,train=True),batch_size=BATCH_SIZE,shuffle=True,pin_memory=PIN)
    model=FusionNet(N_TAB,N_RAD,use_image,use_tabular,use_radiomics,dann=use_dann,n_sites=N_SITES).to(DEVICE)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    scaler=GradScaler(enabled=(DEVICE=="cuda")); pw=None
    if USE_CLASS_WEIGHTS:
        n_pos=labels[train_idx].sum(); pw=torch.tensor([(len(train_idx)-n_pos)/max(1.0,n_pos)],device=DEVICE)
    crit = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pw) if USE_FOCAL_LOSS else nn.BCEWithLogitsLoss(pos_weight=pw)
    dom_crit = nn.CrossEntropyLoss()
    best_auc,best_state=-1.0,None
    epoch_bar = tqdm(range(EPOCHS), desc=desc, leave=False, disable=not show_progress)
    for ep in epoch_bar:
        model.train(); running_loss=0.0; n_batches=0
        tgt_iter = iter(tgt_loader) if use_dann else None
        lambd = dann_lambda(ep, EPOCHS, DANN_LAMBDA_MAX) if use_dann else 0.0
        for img,tab,rad,y,site in tr:
            img=img.to(DEVICE,non_blocking=True); tab=tab.to(DEVICE,non_blocking=True)
            rad=rad.to(DEVICE,non_blocking=True); y=y.to(DEVICE,non_blocking=True); site=site.to(DEVICE,non_blocking=True)
            opt.zero_grad()
            with autocast(enabled=(DEVICE=="cuda")):
                if use_dann:
                    logit, dom_logit_src = model(img,tab,rad,lambd=lambd,return_domain=True)
                    loss = crit(logit, y)
                    dom_loss = dom_crit(dom_logit_src, site)
                    try: t_img,t_tab,t_rad,_,t_site = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(tgt_loader); t_img,t_tab,t_rad,_,t_site = next(tgt_iter)
                    t_img=t_img.to(DEVICE,non_blocking=True); t_tab=t_tab.to(DEVICE,non_blocking=True)
                    t_rad=t_rad.to(DEVICE,non_blocking=True); t_site=t_site.to(DEVICE,non_blocking=True)
                    _, dom_logit_tgt = model(t_img,t_tab,t_rad,lambd=lambd,return_domain=True)
                    dom_loss = dom_loss + dom_crit(dom_logit_tgt, t_site)
                    loss = loss + dom_loss
                else:
                    loss=crit(model(img,tab,rad),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running_loss += loss.item(); n_batches += 1
        sched.step()
        p,yt=predict(model,va); auc=roc_auc_score(yt,p) if len(np.unique(yt))>1 else 0.5
        if auc>best_auc: best_auc=auc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        epoch_bar.set_postfix(loss=f"{running_loss/max(1,n_batches):.3f}", val_auc=f"{auc:.3f}", best=f"{best_auc:.3f}",
                               lambd=f"{lambd:.2f}" if use_dann else "-")
    model.load_state_dict(best_state); return model

print(">>>> 19. Repeated CV (mixed sites, now 3-way) — single run per fold")

def stratifier():
    return (cohort["label"].astype(str)+"_"+cohort["Patient affiliation"].astype(str)).values

def run_cv(use_image,use_tabular,use_radiomics,seed,tag="cv",outer_bar=None):
    global RAD_MAT
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=seed); fa=[];fp=[];oof_p=np.zeros(len(cohort))
    folds=list(skf.split(np.arange(len(cohort)),stratifier()))
    for fi,(tr,va) in enumerate(folds):
        if HARMONIZE_RADIOMICS: RAD_MAT[:] = harmonize_radiomics(rad_mat_raw, tr)
        m=train_one(tr,va,use_image,use_tabular,use_radiomics, use_dann=False,
                    show_progress=True, desc=f"{tag} seed{seed} fold{fi+1}/{len(folds)}")
        p,yt=predict(m,DataLoader(EGFRDataset(va),batch_size=BATCH_SIZE,pin_memory=PIN))
        oof_p[va]=p
        if len(np.unique(yt))>1: fa.append(roc_auc_score(yt,p)); fp.append(average_precision_score(yt,p))
        if outer_bar is not None:
            outer_bar.update(1); outer_bar.set_postfix(fold_auc=f"{fa[-1]:.3f}" if fa else "-")
    return fa,fp,oof_p

def run_cv_repeated(use_image,use_tabular,use_radiomics,tag,n_repeats=N_REPEATS):
    A=[];P=[];OOF=None
    with tqdm(total=n_repeats*N_FOLDS, desc=f"[{tag}] repeats x folds") as outer_bar:
        for r in range(n_repeats):
            fa,fp,oof=run_cv(use_image,use_tabular,use_radiomics,seed=SEED+r,tag=tag,outer_bar=outer_bar); A+=fa;P+=fp
            OOF = oof if OOF is None else OOF
    print(f"[{tag}] AUROC {np.mean(A):.3f} +/- {np.std(A):.3f} | AUPRC {np.mean(P):.3f} +/- {np.std(P):.3f}")
    return A,P,OOF

def logreg_repeated(X,name,n_repeats=N_REPEATS):
    A=[];P=[];OOF=None
    for r in tqdm(range(n_repeats), desc=f"[{name}] repeats", leave=False):
        oof=np.zeros(len(cohort))
        skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED+r)
        for tr,va in skf.split(np.arange(len(cohort)),stratifier()):
            lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
            lr.fit(X[tr],labels[tr].astype(int)); p=lr.predict_proba(X[va])[:,1]; yt=labels[va].astype(int)
            oof[va]=p
            if len(np.unique(yt))>1: A.append(roc_auc_score(yt,p)); P.append(average_precision_score(yt,p))
        if OOF is None: OOF=oof
    print(f"[{name}] AUROC {np.mean(A):.3f} +/- {np.std(A):.3f} | AUPRC {np.mean(P):.3f} +/- {np.std(P):.3f}")
    return A,P,OOF

print(">>> Instant logreg ablations")
A_clin,_,OOF_clin=logreg_repeated(tab_mat,"clinical-only (logreg)")
A_rad,_,_ =logreg_repeated(rad_mat_raw,"radiomics-only (logreg)")
A_cr,_,_  =logreg_repeated(np.hstack([tab_mat,rad_mat_raw]),"clinical+radiomics (logreg)")
print("\n>>> Deep ablations (single run per fold, v3.6: RotNet-pretrained + multi-slice + gated fusion)")
A_cd,_,_    =run_cv_repeated(False,True,False,"clinical-only (deep)")
A_img,_,_   =run_cv_repeated(True,False,False,"image-only (deep)")
A_fus,_,_   =run_cv_repeated(True,True,False,"image+clinical (fusion)")
A_full,_,OOF_full=run_cv_repeated(True,True,True,"image+clinical+radiomics (FULL)")

print(">>>> 20. Stacked ensemble (clinical-logreg + deep-FULL)")

stack_X = np.column_stack([OOF_clin, OOF_full])
skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
stack_auc=[]
for tr,va in skf.split(stack_X, labels.astype(int)):
    meta=LogisticRegression(max_iter=1000)
    meta.fit(stack_X[tr], labels[tr].astype(int))
    p=meta.predict_proba(stack_X[va])[:,1]
    if len(np.unique(labels[va]))>1: stack_auc.append(roc_auc_score(labels[va], p))
print(f"[stacked: clinical + FULL] AUROC {np.mean(stack_auc):.3f} +/- {np.std(stack_auc):.3f}")
print(f"  (for reference) clinical-only mean AUROC over same OOF: {roc_auc_score(labels, OOF_clin):.3f}")
print(f"  (for reference) FULL mean AUROC over same OOF:          {roc_auc_score(labels, OOF_full):.3f}")

print(">>>> 21. Leave-one-site-out — 3-way, single run, real multi-domain DANN")

def loso_logreg(X, name):
    sites=sorted(cohort["Patient affiliation"].astype(str).unique())
    aucs=[]
    for s in sites:
        te=np.where(cohort["Patient affiliation"].astype(str).values==s)[0]
        tr=np.where(cohort["Patient affiliation"].astype(str).values!=s)[0]
        yte=labels[te].astype(int)
        if len(np.unique(yte))<2: continue
        lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
        lr.fit(X[tr],labels[tr].astype(int)); p=lr.predict_proba(X[te])[:,1]
        a=roc_auc_score(yte,p); aucs.append(a)
        print(f"  [{name}] hold-out {s}: AUROC {a:.3f} (n={len(te)}, pos={int(yte.sum())})")
    print(f"  [{name}] LOSO macro-AUROC {np.mean(aucs):.3f}\n"); return np.mean(aucs)

def loso_deep(use_image,use_tabular,use_radiomics,name,use_dann):
    """Single deterministic run per held-out site (no seed averaging, per request)."""
    global RAD_MAT
    sites=sorted(cohort["Patient affiliation"].astype(str).unique()); macro=[]
    tag = f"{name}{' +DANN' if use_dann else ''}"
    for s in tqdm(sites, desc=f"[{tag}] sites", leave=False):
        te=np.where(cohort["Patient affiliation"].astype(str).values==s)[0]
        trall=np.where(cohort["Patient affiliation"].astype(str).values!=s)[0]
        yte=labels[te].astype(int)
        if len(np.unique(yte))<2: continue
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        tr,vsub=train_test_split(trall,test_size=0.2,stratify=labels[trall],random_state=SEED)
        if use_radiomics and HARMONIZE_RADIOMICS: RAD_MAT[:] = harmonize_radiomics(rad_mat_raw, tr)
        m=train_one(tr,vsub,use_image,use_tabular,use_radiomics,
                    target_idx=(te if use_dann else None), use_dann=use_dann,
                    show_progress=True, desc=f"{tag} hold-out {s}")
        p,_=predict(m,DataLoader(EGFRDataset(te),batch_size=BATCH_SIZE,pin_memory=PIN))
        auc=roc_auc_score(yte,p)
        print(f"  [{name}] hold-out {s}: AUROC {auc:.3f} (n={len(te)}, pos={int(yte.sum())}, "
              f"train sites={sorted(cohort.loc[trall,'Patient affiliation'].unique())})")
        macro.append(auc)
    print(f"  [{name}] LOSO macro-AUROC {np.mean(macro):.3f}\n"); return np.mean(macro)

print(">>> LOSO logreg (reference)")
loso_logreg(tab_mat,"clinical-only (logreg)")
loso_logreg(np.hstack([tab_mat,rad_mat_raw]),"clinical+radiomics (logreg)")

print(">>> LOSO deep - PLAIN (no domain adaptation)")
loso_deep(True,False,False,"image-only (plain)", use_dann=False)
loso_deep(True,True,True,"FULL (plain)", use_dann=False)

print(">>> LOSO deep - WITH DANN (2 labeled source domains + held-out unlabeled images)")
loso_deep(True,False,False,"image-only (+DANN)", use_dann=True)
loso_deep(True,True,True,"FULL (+DANN)", use_dann=True)

print(">>>> 22. Save FULL model + single-patient inference")

RAD_MAT[:]=harmonize_radiomics(rad_mat_raw, np.arange(len(cohort)))
all_idx=np.arange(len(cohort))
final_model=train_one(all_idx,all_idx[:max(2,len(all_idx)//5)],True,True,True, use_dann=False, desc="final model")
torch.save(final_model.state_dict(),OUTPUT_DIR/"egfr_fusion_v36.pt"); print("saved")

@torch.no_grad()
def predict_patient(case_id):
    i=cohort.index[cohort["Case ID"]==case_id][0]
    img=torch.from_numpy(np.load(CACHE_DIR/f"{case_id}.npy")).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    tab=torch.from_numpy(tab_mat[i]).float().unsqueeze(0).to(DEVICE)
    rad=torch.from_numpy(RAD_MAT[i]).float().unsqueeze(0).to(DEVICE)
    final_model.eval()
    return {"Case ID":case_id,"P(EGFR-mutant)":round(torch.sigmoid(final_model(img,tab,rad)).item(),3),
            "true_label":POS_LABEL if labels[i]==1 else NEG_LABEL, "site":cohort.iloc[i]["Patient affiliation"]}
print(predict_patient(cohort.iloc[0]["Case ID"]))

print(">>>> 23. Reading the results")



print(f"\n========== RUN COMPLETE {datetime.datetime.now().isoformat()} ==========")
