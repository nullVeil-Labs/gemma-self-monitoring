
"""Gemma self-monitoring pilot. One entry point: plan, run, analyze."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols/pilot-001.json"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")

def validate_protocol(p):
    assert p["id"] == "pilot-001"
    assert p["module"] == "model.language_model.layers.22"
    assert p["strengths"] == [0.05, 0.15, 0.30]
    assert p["conditions"] == ["sham", "concept", "random", "negative"]
    assert len({x["id"] for x in p["concepts"]}) == 4
    assert len({x["id"] for x in p["carriers"]}) == 6
    assert p["label_mappings"] == [0, 1]
    for split, count in (("calibration", 2), ("evaluation", 4)):
        assert len([x for x in p["carriers"] if x["split"] == split]) == count
        assert len([x for x in p["concepts"] if x["split"] == split]) == 2
    return p

def prompt(p, concept, carrier, task, mapping):
    if task == "reading":
        question = carrier["check"]
    else:
        question = p["prompts"][task].format(topic=concept["topic"])
    meanings = ([concept["associate"], concept["foil"]] if task == "association"
                else p["labels"]["binary"])
    if mapping:
        meanings = meanings[::-1]
    legend = f"A = {meanings[0]}\nB = {meanings[1]}"
    return (p["prompts"]["prefix"].format(sentence=carrier["sentence"]) + question
            + p["prompts"]["instruction"].format(legend=legend))

def trial_key(r):
    return (r["concept"], r["carrier"], r["task"], r["mapping"])

def trial_id(r):
    return (*trial_key(r), r["condition"], r["strength"])

def make_trials(p, split, strength=None):
    concepts = [c for c in p["concepts"] if c["split"] == split]
    carriers = [c for c in p["carriers"] if c["split"] == split]
    strengths = p["strengths"] if split == "calibration" else [strength]
    assert all(s in p["strengths"] for s in strengths)
    rows = []
    for c, carrier, task, mapping in itertools.product(
            concepts, carriers, p[split + "_tasks"], p["label_mappings"]):
        common = {"split": split, "concept": c["id"], "carrier": carrier["id"],
                  "task": task, "mapping": mapping,
                  "prompt": prompt(p, c, carrier, task, mapping),
                  "sentence": carrier["sentence"], "reading_answer": carrier["answer"]}
        rows.append(dict(common, condition="sham", strength=0))
        for s, condition in itertools.product(strengths, p["conditions"][1:]):
            rows.append(dict(common, condition=condition, strength=s))
    random.Random(p["seed"] + (split == "evaluation")).shuffle(rows)
    return rows

def check_rows(rows, expected):
    expected_map = {trial_id(r): r for r in expected}
    if len(rows) != len(expected_map) or {trial_id(r) for r in rows} != set(expected_map):
        raise ValueError("Missing or duplicate trials; no partial analysis is allowed.")
    for row in rows:
        for k, v in expected_map[trial_id(row)].items():
            if row[k] != v:
                raise ValueError("Trial differs from frozen design: " + k)
        for k in ("p_positive", "label_mass"):
            if not math.isfinite(row[k]) or not 0 <= row[k] <= 1:
                raise ValueError("Invalid probability.")
        if row["decision"] not in ("positive", "negative", "invalid"):
            raise ValueError("Unknown report label.")

def mean(values):
    return statistics.mean(values)

def quality(rows):
    reading = [r for r in rows if r["task"] == "reading"]
    return {
        "valid_label_rate": mean(r["decision"] != "invalid" for r in rows),
        "mean_label_mass": mean(r["label_mass"] for r in rows),
        "reading_accuracy": mean(r["decision"] != "invalid" and
                                 (r["decision"] == "positive") == r["reading_answer"] for r in reading),
    }

def gain(rows, baseline, task):
    base = {trial_key(r): r for r in baseline}
    return mean(r["p_positive"] - base[trial_key(r)]["p_positive"]
                for r in rows if r["task"] == task)

def calibration_summary(rows, p):
    check_rows(rows, make_trials(p, "calibration"))
    base = [r for r in rows if r["condition"] == "sham"]
    bq, gate = quality(base), p["calibration_gate"]
    def acceptable(q):
        return (q["reading_accuracy"] >= gate["min_reading_accuracy"]
                and q["valid_label_rate"] >= gate["min_valid_label_rate"]
                and q["mean_label_mass"] >= gate["min_mean_label_mass"])
    candidates = []
    for s in p["strengths"]:
        group = [r for r in rows if r["condition"] == "concept" and r["strength"] == s]
        q = quality(group)
        g = gain(group, base, "association")
        candidates.append({"strength": s, **q, "association_probability_gain": g,
                           "passes": acceptable(bq) and acceptable(q)
                           and g >= gate["min_association_probability_gain"]})
    selected = next((x["strength"] for x in candidates if x["passes"]), None)
    controls = []
    for s, condition in itertools.product(p["strengths"], ("random", "negative")):
        group = [r for r in rows if r["strength"] == s and r["condition"] == condition]
        controls.append({"strength": s, "condition": condition, **quality(group),
                         "association_probability_gain": gain(group, base, "association")})
    return {"baseline": bq, "candidates": candidates, "controls": controls,
            "selected_strength": selected,
            "decision": "Proceed to held-out evaluation" if selected is not None
            else "Stop: calibration did not establish a usable, format-preserving perturbation."}

def evaluation_summary(rows, p, strength):
    check_rows(rows, make_trials(p, "evaluation", strength))
    base = [r for r in rows if r["condition"] == "sham"]
    groups = {}
    for condition in p["conditions"]:
        rr = [r for r in rows if r["condition"] == condition]
        det = [r for r in rr if r["task"] == "detection"]
        ext = [r for r in rr if r["task"] == "external"]
        groups[condition] = dict(quality(rr),
            detection_positive_rate=mean(r["decision"] == "positive" for r in det),
            detection_probability_gain=gain(rr, base, "detection"),
            detection_by_mapping={str(m): mean(r["decision"] == "positive" for r in det if r["mapping"] == m) for m in (0, 1)},
            external_positive_rate=mean(r["decision"] == "positive" for r in ext),
            external_probability_gain=gain(rr, base, "external"),
            association_probability_gain=gain(rr, base, "association"))
    c, b, gate = groups["concept"], groups["sham"], p["evaluation_gate"]
    passed = (c["detection_positive_rate"] >= gate["min_detection_hit_rate"]
        and b["detection_positive_rate"] <= gate["max_sham_false_positive_rate"]
        and min(c["detection_by_mapping"].values()) >= gate["min_mapping_hit_rate"]
        and c["external_positive_rate"] <= gate["max_external_false_positive_rate"]
        and c["external_probability_gain"] <= gate["max_external_probability_gain"]
        and c["association_probability_gain"] >= gate["min_association_probability_gain"]
        and all(x["reading_accuracy"] >= gate["min_reading_accuracy"] and
                x["valid_label_rate"] >= gate["min_valid_label_rate"] and
                x["mean_label_mass"] >= gate["min_mean_label_mass"] for x in (c, b)))
    return {"conditions": groups, "passes_pilot_gates": passed,
            "interpretation": "Candidate behavior requiring independent replication; no self-awareness claim."
            if passed else "This pilot did not establish controlled perturbation detection.",
            "inference": "Descriptive only; repeated shams and label reversals are not independent samples."}

def verify_runtime(resources):
    vet = read(resources["previous_runtime_inventory"])
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(vet["checked_at"])).total_seconds()
    if not vet.get("approved") or not 0 <= age <= 72 * 3600:
        raise ValueError("Runtime review must be current and approved.")
    if Path(sys.prefix).resolve() != Path(vet["python_prefix"]).resolve():
        raise ValueError("Use the previously reviewed isolated interpreter.")
    normalize = lambda s: s.lower().replace("_", "-").replace(".", "-")
    actual = {}
    for d in importlib.metadata.distributions():
        name = normalize(d.metadata["Name"])
        if name in actual:
            raise ValueError("Duplicate runtime distribution.")
        actual[name] = d.version
    if actual != vet["distributions"] or vet["known_advisories"] != []:
        raise ValueError("Runtime inventory or advisory check failed.")
    if not vet["package_integrity_verified"] or vet["minimum_release_age_days"] < 14:
        raise ValueError("Recorded package review requirements are unmet.")
    model = Path(resources["model_directory"]).resolve()
    if str(model).lower() != str(Path(vet["model_dir"]).resolve()).lower():
        raise ValueError("Checkpoint differs from user-confirmed path.")
    if vet["model_hash_policy"] != "skipped_at_user_request":
        raise ValueError("Model hashing must remain skipped.")
    inventory = {}
    for f in model.rglob("*"):
        if f.is_symlink():
            raise ValueError("Unexpected model symlink.")
        if f.is_file():
            if f.suffix not in (".json", ".safetensors", ".model", ".txt", ".md"):
                raise ValueError("Unexpected non-data model file.")
            inventory[f.relative_to(model).as_posix()] = {"size_bytes": f.stat().st_size}
    if inventory != vet["model_files"]:
        raise ValueError("Model file names or sizes changed.")
    config = read(model / "config.json")
    if config.get("auto_map") or config.get("quantization_config") or config["architectures"] != ["Gemma3ForConditionalGeneration"]:
        raise ValueError("Unsupported/custom checkpoint configuration.")
    if config["text_config"]["num_hidden_layers"] != 34:
        raise ValueError("Unexpected layer count.")
    for shard in read(model / "model.safetensors.index.json")["weight_map"].values():
        if shard not in inventory or not shard.endswith(".safetensors"):
            raise ValueError("Invalid checkpoint shard reference.")
    return vet, model


def perturb_tensor(h, positions, direction, strength):
    """Return a copy changed only at the carrier positions, plus actual norm ratio."""
    original = h[:, positions, :].float()
    delta = strength * original.norm(dim=-1, keepdim=True) * direction
    modified = h.clone()
    modified[:, positions, :] = (original + delta).to(h.dtype)
    ratio = ((modified[:, positions, :].float() - original).norm(dim=-1)
             / original.norm(dim=-1).clamp_min(1e-12)).mean().item()
    return modified, ratio

def execute(p, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Use a simple unique run ID.")
    resources = read(ROOT / "local-resources.json")
    vet, model_path = verify_runtime(resources)
    folder = ROOT / "results" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": p, "protocol_sha256": fingerprint(PROTOCOL),
                "runner_sha256": fingerprint(Path(__file__)),
                "resources": resources, "runtime_review": vet}
    save(folder / "manifest.json", manifest)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    import torch
    import numpy as np
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 is required.")
    def seed():
        random.seed(p["seed"]); np.random.seed(p["seed"]); torch.manual_seed(p["seed"])
        torch.cuda.manual_seed_all(p["seed"])
    seed()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    block = dict(model.named_modules())[p["module"]]
    save(folder / "runtime.json", {"python": sys.version, "torch": torch.__version__,
         "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
         "attention": model.config.text_config._attn_implementation,
         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
    label_ids = {}
    for label, spellings in p["labels"]["token_spellings"].items():
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in spellings]
        label_ids[label] = sorted({x[0] for x in ids if len(x) == 1})
        if not label_ids[label]:
            raise ValueError("No single-token answer label.")
    if set(label_ids["A"]) & set(label_ids["B"]):
        raise ValueError("Overlapping label token sets.")
    directions, randoms, donor_records = {}, {}, []
    def donor(word):
        text = p["donor_template"].format(word=word)
        ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        captured = []
        def capture(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0, -1].detach().float().cpu())
        handle = block.register_forward_hook(capture)
        try:
            seed()
            with torch.no_grad():
                model(ids, use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("Donor hook did not fire exactly once.")
        donor_records.append({"word": word, "text": text, "token_ids": ids[0].tolist()})
        return captured[0]
    neutral = torch.stack([donor(w) for w in p["neutral_words"]]).mean(0)
    def prepare(split):
        for index, c in enumerate(p["concepts"]):
            if c["split"] != split:
                continue
            d = torch.stack([donor(w) for w in c["words"]]).mean(0) - neutral
            if not torch.isfinite(d).all() or d.norm() < 1e-8:
                raise ValueError("Invalid concept direction.")
            d = d / d.norm()
            rng = torch.Generator().manual_seed(p["seed"] + index)
            v = torch.randn(d.shape, generator=rng)
            v = v - (v @ d) * d
            v = v / v.norm()
            directions[c["id"]], randoms[c["id"]] = d.to("cuda"), v.to("cuda")
        save(folder / (split + "-directions.json"), {
             c["id"]: {"concept": directions[c["id"]].cpu().tolist(),
                       "random": randoms[c["id"]].cpu().tolist()}
             for c in p["concepts"] if c["split"] == split})
        save(folder / (split + "-donors.json"), donor_records)
    def measure(trial):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": trial["prompt"]}], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        marker = "<sentence>" + trial["sentence"] + "</sentence>"
        if rendered.count(marker) != 1:
            raise ValueError("Carrier marker must occur exactly once.")
        start = rendered.index(marker) + len("<sentence>")
        end = start + len(trial["sentence"])
        positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > start and a < end]
        if not positions or max(positions) >= len(encoded["input_ids"]) - 1:
            raise ValueError("Invalid intervention span.")
        ids = torch.tensor([encoded["input_ids"]], device="cuda")
        direction = directions[trial["concept"]]
        if trial["condition"] == "random":
            direction = randoms[trial["concept"]]
        elif trial["condition"] == "negative":
            direction = -direction
        count, actual_ratios = [], []
        def inject(_m, _i, output):
            count.append(1)
            h = output[0] if isinstance(output, tuple) else output
            if trial["condition"] == "sham":
                return output
            modified, ratio = perturb_tensor(h, positions, direction, trial["strength"])
            actual_ratios.append(ratio)
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        handle = block.register_forward_hook(inject)
        try:
            seed()
            with torch.no_grad():
                logits = model(ids, use_cache=False).logits[0, -1].float()
        finally:
            handle.remove()
        if len(count) != 1 or block._forward_hooks:
            raise ValueError("Hook invocation/cleanup invariant failed.")
        probabilities = torch.softmax(logits, dim=-1)
        pa = probabilities[label_ids["A"]].sum().item()
        pb = probabilities[label_ids["B"]].sum().item()
        mass = pa + pb
        if mass < 1e-30:
            raise ValueError("Answer label probabilities underflowed.")
        top = logits.argmax().item()
        label = "A" if top in label_ids["A"] else "B" if top in label_ids["B"] else None
        positive_label = "B" if trial["mapping"] else "A"
        return dict(trial, token_ids=encoded["input_ids"], rendered_prompt=rendered,
                    injection_positions=positions, actual_norm_ratio=actual_ratios[0] if actual_ratios else 0,
                    label_ids=label_ids, p_A=pa, p_B=pb,
                    p_positive=(pb if trial["mapping"] else pa) / mass,
                    label_mass=min(mass, 1.0), greedy_token_id=top,
                    greedy_token=tokenizer.decode([top]),
                    decision="invalid" if label is None else "positive" if label == positive_label else "negative")
    def stage(split, strength=None):
        prepare(split)
        trials = make_trials(p, split, strength)
        save(folder / (split + "-schedule.json"), trials)
        rows = []
        start = time.monotonic()
        with (folder / (split + ".jsonl")).open("x", encoding="utf-8") as stream:
            for trial in trials:
                row = measure(trial)
                stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
                rows.append(row)
                if len(rows) % 10 == 0:
                    print(f"{split}: {len(rows)}/{len(trials)} ({time.monotonic()-start:.0f}s)", flush=True)
        return rows
    calibration = calibration_summary(stage("calibration"), p)
    save(folder / "calibration-summary.json", calibration)
    print(json.dumps(calibration, indent=2), flush=True)
    if calibration["selected_strength"] is None:
        save(folder / "completion.json", {"status": "stopped_at_calibration",
             "evaluation_run": False, "reason": calibration["decision"]})
        return
    strength = calibration["selected_strength"]
    evaluation = evaluation_summary(stage("evaluation", strength), p, strength)
    save(folder / "evaluation-summary.json", evaluation)
    save(folder / "completion.json", {"status": "completed", "evaluation_run": True,
         "passes_pilot_gates": evaluation["passes_pilot_gates"]})
    print(json.dumps(evaluation, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyze"), nargs="?", default="plan")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    p = validate_protocol(read(PROTOCOL))
    if args.command == "plan":
        print(json.dumps({"id": p["id"], "protocol_sha256": fingerprint(PROTOCOL),
             "calibration_trials": len(make_trials(p, "calibration")),
             "evaluation_trials_if_calibration_passes": len(make_trials(p, "evaluation", p["strengths"][0])),
             "donor_forwards": {"calibration": 12, "evaluation_if_run": 8},
             "status": "Plan only; no ML imports or model access."}, indent=2))
    elif args.command == "run":
        execute(p, args.run_id)
    else:
        if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
            raise ValueError("A simple run ID is required.")
        folder = ROOT / "results" / args.run_id
        manifest = read(folder / "manifest.json")
        if manifest["protocol"] != p or manifest["protocol_sha256"] != fingerprint(PROTOCOL):
            raise ValueError("Frozen protocol mismatch.")
        rows = [json.loads(line) for line in (folder / "calibration.jsonl").read_text(encoding="utf-8").splitlines()]
        result = calibration_summary(rows, p)
        print(json.dumps(result, indent=2))
        if result["selected_strength"] is not None and (folder / "evaluation.jsonl").exists():
            rows = [json.loads(line) for line in (folder / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
            print(json.dumps(evaluation_summary(rows, p, result["selected_strength"]), indent=2))

if __name__ == "__main__":
    main()
