import argparse
import json
import os
import random
import re
import time

import evaluate
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import unimernet.tasks as tasks
from unimernet.common.config import Config
from unimernet.datasets.builders import *  # noqa: F401,F403
from unimernet.models import *  # noqa: F401,F403
from unimernet.processors import *  # noqa: F401,F403
from unimernet.processors import load_processor
from unimernet.tasks import *  # noqa: F401,F403


SET_MAP = {
    "spe": "Simple Print Expression (SPE)",
    "cpe": "Complex Print Expression (CPE)",
    "sce": "Screen Capture Expression (SCE)",
    "hwe": "Handwritten Expression (HWE)",
}


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(raw_image) if self.transform else raw_image
        return image


def load_data(image_dir, math_file):
    """
    Load image paths and corresponding formulas.
    Assumes image names are 0000000.png, 0000001.png, ...
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isfile(math_file):
        raise FileNotFoundError(f"Annotation file not found: {math_file}")

    image_names = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    )
    image_name_set = set(image_names)

    image_paths = []
    math_gts = []

    with open(math_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            image_name = f"{i:07d}.png"
            gt = line.strip()

            if not gt:
                continue
            if image_name not in image_name_set:
                continue

            image_paths.append(os.path.join(image_dir, image_name))
            math_gts.append(gt)

    if not image_paths:
        raise ValueError(f"No valid samples found in {image_dir} with annotation {math_file}")

    if len(image_paths) != len(math_gts):
        raise ValueError("The number of images does not match the number of formulas.")

    return image_paths, math_gts


def normalize_text(text):
    """
    Remove unnecessary whitespace from LaTeX code.
    """
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = r"[a-zA-Z]"
    noletter = r"[\W_^\d]"

    names = [x[0].replace(" ", "") for x in re.findall(text_reg, text)]
    text = re.sub(text_reg, lambda match: str(names.pop(0)), text)

    news = text
    while True:
        text = news
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", text)
        news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
        news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
        if news == text:
            break
    return text


def score_text(predictions, references):
    bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, int(1e8)))
    bleu_results = bleu.compute(predictions=predictions, references=references)

    lev_dist = []
    exact_matches = 0

    for pred, ref in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(pred, ref))
        if pred == ref:
            exact_matches += 1

    total = len(references)
    exprate = exact_matches / total if total > 0 else 0.0
    avg_edit = sum(lev_dist) / len(lev_dist) if lev_dist else 0.0

    return {
        "bleu": bleu_results["bleu"],
        "edit": avg_edit,
        "exact_match": exact_matches,
        "exprate": exprate,
        "total": total,
    }


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Test UniMERNet on one subset at a time.")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument(
        "--set",
        required=True,
        choices=["spe", "cpe", "sce", "hwe"],
        help="Which test subset to evaluate.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="Optional path to save detailed prediction results as txt.",
    )
    parser.add_argument(
        "--cdm_json_path",
        type=str,
        default=None,
        help="Optional path to save predictions in CDM json format.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from YAML.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override num_workers from YAML.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "Override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file."
        ),
    )
    return parser.parse_args()


def resolve_subset_paths(cfg, subset):
    base_dir = cfg.config.datasets.formula_rec_eval.build_info.base_dir
    image_dir = os.path.join(base_dir, subset)
    math_file = os.path.join(base_dir, f"{subset}.txt")
    return image_dir, math_file


def save_results(result_path, subset_name, predictions, references):
    os.makedirs(os.path.dirname(result_path), exist_ok=True) if os.path.dirname(result_path) else None
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Subset: {subset_name}\n\n")
        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"[{idx}]\n")
            f.write(f"PRED: {pred}\n")
            f.write(f"REF : {ref}\n\n")


def save_cdm_json(save_path, image_list, predictions, references):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    items = []
    for idx, (img_path, pred, ref) in enumerate(zip(image_list, predictions, references)):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        items.append({
            "img_id": img_id if img_id else f"case_{idx}",
            "gt": ref,
            "pred": pred,
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    cfg = Config(args)

    seed = getattr(cfg.config.run, "seed", 42)
    setup_seeds(seed)

    start = time.time()

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)

    use_cuda = torch.cuda.is_available() and str(cfg.config.run.device).lower() == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    vis_processor = load_processor(
        "formula_image_eval",
        cfg.config.datasets.formula_rec_eval.vis_processor.eval,
    )

    model.to(device)
    model.eval()

    load_done = time.time()

    subset = args.set.lower()
    subset_name = SET_MAP[subset]
    image_dir, math_file = resolve_subset_paths(cfg, subset)

    image_list, math_gts = load_data(image_dir, math_file)

    transform = transforms.Compose([vis_processor])
    dataset = MathDataset(image_list, transform=transform)

    batch_size = args.batch_size or int(getattr(cfg.config.run, "batch_size_eval", 4))
    num_workers = args.num_workers if args.num_workers is not None else int(getattr(cfg.config.run, "num_workers", 2))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_cuda,
    )

    print("CFG PATH:", args.cfg_path)
    print("MODEL NAME:", cfg.config.model.model_name)
    print("TOKENIZER PATH:", cfg.config.model.tokenizer_config.path)
    print("FINETUNED:", cfg.config.model.finetuned)
    print(f"arch_name: {cfg.config.model.arch}")
    print(f"model_type: {cfg.config.model.model_type}")
    print(f"checkpoint: {cfg.config.model.finetuned}")
    print("=" * 100)
    print(f"Device: {device}")
    print(f"Load model: {load_done - start:.3f}s")
    print(f"Subset: {subset_name}")
    print(f"Image dir: {image_dir}")
    print(f"Annotation: {math_file}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print("=" * 100)

    infer_start = time.time()
    math_preds = []

    for images in tqdm(dataloader, desc=f"Testing {subset}"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            output = model.generate({"image": images})
        math_preds.extend(output["pred_str"])

    infer_end = time.time()

    norm_gts = [normalize_text(gt) for gt in math_gts]
    norm_preds = [normalize_text(pred) for pred in math_preds]

    print(f"len_gts: {len(norm_gts)}, len_preds: {len(norm_preds)}")
    if norm_gts:
        print(f"norm_gts[0]: {norm_gts[0]}")
    if norm_preds:
        print(f"norm_preds[0]: {norm_preds[0]}")

    scores = score_text(norm_preds, norm_gts)

    score_headers = ["bleu ↑", "edit ↓", "exact_match ↑", "exprate ↑"]
    score_table = [[
        scores["bleu"],
        scores["edit"],
        scores["exact_match"],
        scores["exprate"],
    ]]

    print(f"Evaluation Set: {subset_name}")
    print(f"Inference Time: {infer_end - infer_start:.3f}s")
    print(tabulate(score_table, headers=score_headers, floatfmt=".6f"))
    print(f"Exact Match: {scores['exact_match']}/{scores['total']}")
    print(f"ExpRate: {scores['exprate']:.4%}")
    print("=" * 100)

    if args.result_path:
        save_results(args.result_path, subset_name, norm_preds, norm_gts)
        print(f"Saved detailed results to: {args.result_path}")

    if args.cdm_json_path:
        save_cdm_json(args.cdm_json_path, image_list, norm_preds, norm_gts)
        print(f"Saved CDM json to: {args.cdm_json_path}")


if __name__ == "__main__":
    main()
