"""Public detector weights used by the video work: where they come from, their hash and licence.

Nothing here is ours or private; the organisers can download every file from the URL listed.
Weights land in $MTAH_VIDEO_MODELS (default ~/.cache/wt-mtah-video/models), outside the repo.

    python -m mtah.video.weights          # download missing files, verify every sha256

Every checkpoint is a torch pickle. Before first load each one was scanned statically
(pickletools, nothing executed) for the globals it imports: all five reference only torch,
collections, ultralytics and builtins.set. `scan()` repeats that check.
"""

import hashlib
import os
import pickletools
import urllib.request
import zipfile

MODELS_DIR = os.environ.get("MTAH_VIDEO_MODELS") or os.path.expanduser(
    "~/.cache/wt-mtah-video/models")

HF = "https://huggingface.co/{}/resolve/main/{}"
UL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"

WEIGHTS = {
    "coco11n": dict(path="coco/yolo11n.pt", url=UL.format("yolo11n.pt"),
                    sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
                    licence="AGPL-3.0 (Ultralytics YOLO11, COCO)"),
    "coco11s": dict(path="coco/yolo11s.pt", url=UL.format("yolo11s.pt"),
                    sha256="85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
                    licence="AGPL-3.0 (Ultralytics YOLO11, COCO)"),
    "soccana": dict(path="soccana/best.pt",
                    url=HF.format("Adit-jain/soccana", "Model/weights/best.pt"),
                    sha256="dd5f0bec29931dfe22f6dc691dde75d5cf3d3b48ec4cfe5ac2d115538cfd1123",
                    licence="no model-card licence; checkpoint metadata says AGPL-3.0 (Ultralytics)"),
    "acatorcini": dict(path="acatorcini/v9b_best.pt",
                       url=HF.format("acatorcini/yolov9-soccer-ball", "v9b_best.pt"),
                       sha256="f953fd3114377265ed872de3d6ebe613647bc48aa858258a68ee1a19f4682b73",
                       licence="AGPL-3.0 (model card)"),
    "uisikdag": dict(path="uisikdag/best.pt",
                     url=HF.format("uisikdag/yolo-v8-football-players-detection", "best.pt"),
                     sha256="844e9ad1ebafbd0c07f066e45aa87d9b87595dd9092dfee9cac3b10a6a5fa730",
                     licence="none stated (model card or checkpoint); Ultralytics YOLOv8 is AGPL-3.0"),
}

ALLOWED_PICKLE_MODULES = {"torch", "collections", "ultralytics", "builtins", "__builtin__"}


def local_path(key):
    return os.path.join(MODELS_DIR, WEIGHTS[key]["path"])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pickle_globals(path):
    """Every module.name a torch zip checkpoint's pickles would import. Executes nothing."""
    found = set()
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not n.endswith(".pkl"):
                continue
            strings = []
            for op, arg, _ in pickletools.genops(z.read(n)):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                    strings.append(arg)
                elif op.name == "GLOBAL":
                    found.add(arg.replace(" ", "."))
                elif op.name == "STACK_GLOBAL":
                    found.add(strings[-2] + "." + strings[-1])
    return found


def scan(path):
    """Raise unless the checkpoint imports only ALLOWED_PICKLE_MODULES (and builtins.set)."""
    bad = sorted(g for g in pickle_globals(path)
                 if g.split(".")[0] not in ALLOWED_PICKLE_MODULES
                 or (g.split(".")[0] in ("builtins", "__builtin__") and g.split(".", 1)[1] != "set"))
    if bad:
        raise ValueError("{} imports unexpected globals: {}".format(path, bad[:10]))


def fetch(key):
    """Local path of a verified, scanned checkpoint; downloads it if missing."""
    w = WEIGHTS[key]
    path = local_path(key)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(w["url"], path + ".part")
        os.replace(path + ".part", path)
    got = sha256(path)
    if got != w["sha256"]:
        raise ValueError("{}: sha256 {} != {}".format(path, got, w["sha256"]))
    scan(path)
    return path


def openvino(key, imgsz=(1088, 1920)):
    """OpenVINO export of a verified checkpoint (fixed input size), made once next to the weights.

    Same network and weights; OpenVINO is only a faster CPU runtime. docs/video_pilot.md compares
    its detections with the torch path on the same frames.
    """
    import shutil
    out = os.path.join(MODELS_DIR, "export", "{}_{}x{}_openvino_model".format(key, *imgsz))
    if not os.path.isdir(out):
        from ultralytics import YOLO
        src = fetch(key)
        tmp = os.path.join(MODELS_DIR, "export", "{}_{}x{}.pt".format(key, *imgsz))
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        shutil.copyfile(src, tmp)
        made = YOLO(tmp).export(format="openvino", imgsz=imgsz, half=False, dynamic=False, batch=1)
        os.replace(made, out)
    return out


if __name__ == "__main__":
    for k in WEIGHTS:
        print(k, fetch(k))
