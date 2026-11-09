
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
import re
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
    if p["id"] == "pilot-005":
        reference = ROOT / "protocols/pilot-004.json"
        p4 = read(reference)
        if fingerprint(reference) != p["readout_reference"]["sha256"] or any(p[k] != p4[k] for k in ("word_banks", "free_templates")):
            raise ValueError("Validated readout changed.")
        if p["sites"] != ["carrier"] or p["strengths"] != [0.05, 0.15, 0.3] or p["evaluation_tasks"]:
            raise ValueError("Carrier-only bounded calibration required.")
        if p["module"] != "model.language_model.layers.22" or p["conditions"] != ["sham", "concept", "random", "negative"]:
            raise ValueError("Unexpected intervention design.")
        if len(p["carriers"]) != 4 or len({c["id"] for c in p["carriers"]}) != 4 or len(p["concepts"]) != 2:
            raise ValueError("Expected four carriers and two concepts.")
        if any(c["split"] != "calibration" for c in p["carriers"] + p["concepts"]):
            raise ValueError("Evaluation material forbidden.")
        return p
    if p["id"] == "pilot-004":
        if p["conditions"] != ["visible"] or p["evaluation_tasks"] != []:
            raise ValueError("Readout validation cannot contain interventions or evaluation.")
        if len(p["controls"]) != 8 or len({c["id"] for c in p["controls"]}) != 8:
            raise ValueError("Expected eight unique control triplets.")
        if set(p["word_banks"]) != {"fruit", "music"} or any(len(set(w)) != 20 for w in p["word_banks"].values()):
            raise ValueError("Expected two fixed 20-word banks.")
        bank_words = [w for words in p["word_banks"].values() for w in words]
        if len(set(bank_words)) != 40 or len(p["free_templates"]) != 2:
            raise ValueError("Overlapping vocabulary or wrong template count.")
        sentences = [c[group] for c in p["controls"] for group in ("fruit", "music", "neutral")]
        if len(set(sentences)) != 24 or any(set(re.findall(r"[a-z]+", text.lower())) & set(bank_words) for text in sentences):
            raise ValueError("Controls must be unique and must not display scoring words.")
        return p
    if p["id"] in ("pilot-002", "pilot-003", "pilot-005"):
        assert p["module"] == "model.language_model.layers.22"
        assert p["strengths"] == [0.05, 0.15, 0.30]
        assert p["sites"] == ["carrier", "response"] and p["evaluation_tasks"] == []
        assert len(p["concepts"]) == len(p["carriers"]) == 2
        assert all(c["split"] == "calibration" for c in p["concepts"] + p["carriers"])
        return p
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
    return (r["concept"], r["carrier"], r["task"], r["mapping"], r.get("template"), r.get("site"))

def trial_id(r):
    return (*trial_key(r), r["condition"], r["strength"])

def make_trials(p, split, strength=None):
    if p["id"] == "pilot-004":
        if split != "calibration" or strength is not None:
            raise ValueError("Visible readout validation has no intervention/evaluation stage.")
        return make_readout_trials(p)
    if p["id"] in ("pilot-002", "pilot-003", "pilot-005"):
        if split != "calibration":
            raise ValueError("pilot-002 has no self-monitoring evaluation.")
        return make_diagnostic_trials(p)
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
    if p["id"] == "pilot-004":
        return visible_readout_summary(rows, p)
    if p["id"] in ("pilot-002", "pilot-003", "pilot-005"):
        return diagnostic_summary(rows, p)
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


def make_diagnostic_trials(p):
    rows = []
    specs = [("free_association", 0, 0), ("free_association", 0, 1),
             ("reading", 0, 0), ("reading", 1, 0)]
    def build(c, carrier, task, mapping, template, site):
        text = (p["prompts"]["prefix"].format(sentence=carrier["sentence"])
                + p["free_templates"][template]) if task == "free_association" else prompt(p, c, carrier, task, mapping)
        return {"split": "calibration", "concept": c["id"], "carrier": carrier["id"],
                "task": task, "mapping": mapping, "template": template, "site": site,
                "prompt": text, "sentence": carrier["sentence"], "reading_answer": carrier["answer"]}
    for c, carrier, site, spec in itertools.product(p["concepts"], p["carriers"], p["sites"], specs):
        task, mapping, template = spec
        common = build(c, carrier, task, mapping, template, site)
        rows.append(dict(common, condition="sham", strength=0))
        for strength, condition in itertools.product(p.get("strengths_by_site", {}).get(site, p["strengths"]), p["conditions"][1:]):
            rows.append(dict(common, condition=condition, strength=strength))
    for c, template in itertools.product(p["concepts"], range(len(p["free_templates"]))):
        carrier = {"id": "visible-" + c["id"], "sentence": p["visible_sentences"][c["id"]], "answer": False}
        rows.append(dict(build(c, carrier, "free_association", 0, template, "visible"),
                         condition="visible", strength=0))
    random.Random(p["seed"]).shuffle(rows)
    if p["id"] in ("pilot-003", "pilot-005"):
        rows.sort(key=lambda r: r["condition"] != "visible")
    return rows


def readout_preflight(rows, p):
    visible = [r for r in rows if r["condition"] == "visible"]
    expected = [r for r in make_diagnostic_trials(p) if r["condition"] == "visible"]
    check_rows(visible, expected)
    controls = {c["id"]: {
        "mean_log_odds": mean(r["family_log_odds"] for r in visible if r["concept"] == c["id"]),
        "mean_family_mass": mean(r["target_mass"] + r["foil_mass"] for r in visible if r["concept"] == c["id"])
    } for c in p["concepts"]}
    gate = p["preflight"]
    passed = all(v["mean_family_mass"] >= gate["min_mean_family_mass"]
                 and v["mean_log_odds"] > gate["min_mean_log_odds"] for v in controls.values())
    return {"controls": controls, "passes": passed}

def diagnostic_summary(rows, p):
    preflight = readout_preflight(rows, p) if p["id"] in ("pilot-003", "pilot-005") else None
    if preflight and len(rows) == 4 and not preflight["passes"]:
        return {"preflight": preflight, "candidates": [],
                "selected_by_site": {site: None for site in p["sites"]},
                "evaluation_run": False, "decision": "Stop: visible readout preflight failed."}
    check_rows(rows, make_diagnostic_trials(p))
    for r in rows:
        if r["task"] == "free_association":
            if not math.isfinite(r["family_log_odds"]):
                raise ValueError("Nonfinite word-bank log odds.")
            for key in ("target_mass", "foil_mass"):
                if not math.isfinite(r[key]) or not 0 <= r[key] <= 1:
                    raise ValueError("Invalid word-bank probability.")
    base = {trial_key(r): r for r in rows if r["condition"] == "sham"}
    gate = p["calibration_gate"]
    candidates = []
    for site, strength in ((site, strength) for site in p["sites"] for strength in p.get("strengths_by_site", {}).get(site, p["strengths"])):
        groups = {condition: [r for r in rows if r["site"] == site and r["strength"] == strength
                              and r["condition"] == condition] for condition in p["conditions"][1:]}
        per_concept = {}
        concept_templates = itertools.product(p["concepts"], range(2) if p["id"] == "pilot-005" else [None])
        for c, template in concept_templates:
            means = {}
            selected = []
            for condition, rr in groups.items():
                selected_rows = [r for r in rr if r["concept"] == c["id"] and r["task"] == "free_association" and (template is None or r["template"] == template)]
                deltas = [r["family_log_odds"] - base[trial_key(r)]["family_log_odds"] for r in selected_rows]
                means[condition] = mean(deltas)
                if condition == "concept":
                    selected = selected_rows
                    positive_fraction = mean(d > 0 for d in deltas)
            per_concept[c["id"] if template is None else f"{c['id']}/template-{template}"] = {
                "log_odds_gains": means, "positive_cell_fraction": positive_fraction,
                "mean_family_mass": mean(r["target_mass"] + r["foil_mass"] for r in selected),
                "mean_target_mass_gain": mean(r["target_mass"] - base[trial_key(r)]["target_mass"] for r in selected)}
        reading = {condition: quality([r for r in rr if r["task"] == "reading"])
                   for condition, rr in groups.items()}
        baseline_reading = quality([r for r in base.values() if r["site"] == site and r["task"] == "reading"])
        def q_pass(q):
            return (q["reading_accuracy"] >= gate["min_reading_accuracy"]
                    and q["valid_label_rate"] >= gate["min_valid_reading_label_rate"]
                    and q["mean_label_mass"] >= gate["min_mean_reading_label_mass"])
        passed = q_pass(reading["concept"]) and q_pass(baseline_reading) and (preflight is None or preflight["passes"])
        if p["id"] == "pilot-005":
            for condition in p["conditions"]:
                source = list(base.values()) if condition == "sham" else groups[condition]
                for c in p["concepts"]:
                    q = quality([r for r in source if r["task"] == "reading" and r["concept"] == c["id"]])
                    reading[f"{condition}/{c['id']}"] = q
                    passed &= q_pass(q)
        for item in per_concept.values():
            gains = item["log_odds_gains"]
            passed &= (gains["concept"] >= gate["min_each_concept_mean_log_odds_gain"]
                       and item["positive_cell_fraction"] >= gate["min_each_concept_positive_cell_fraction"]
                       and item["mean_family_mass"] >= gate["min_each_concept_mean_family_mass"]
                       and gains["concept"] - gains["random"] >= gate["min_concept_minus_random_gain"]
                       and gains["concept"] - gains["negative"] >= gate["min_concept_minus_negative_gain"])
            if p["id"] == "pilot-005":
                passed &= item["mean_target_mass_gain"] >= gate["min_each_concept_template_target_mass_gain"]
        candidates.append({"site": site, "strength": strength, "concepts": per_concept,
                           "reading": reading, "baseline_reading": baseline_reading, "passes": bool(passed)})
    visible = {c["id"]: {
        "mean_log_odds": mean(r["family_log_odds"] for r in rows if r["condition"] == "visible" and r["concept"] == c["id"]),
        "mean_family_mass": mean(r["target_mass"] + r["foil_mass"] for r in rows if r["condition"] == "visible" and r["concept"] == c["id"])
    } for c in p["concepts"]}
    result = {"candidates": candidates, "visible_controls": visible,
            "selected_by_site": {site: next((r["strength"] for r in candidates if r["site"] == site and r["passes"]), None)
                                 for site in p["sites"]},
            "evaluation_run": False, "decision": "Calibration diagnostic only; no self-monitoring verdict."}
    if preflight is not None:
        result["preflight"] = preflight
    return result


def make_readout_trials(p):
    rows = []
    for control, group, template in itertools.product(p["controls"], ("fruit", "music", "neutral"), range(2)):
        sentence = control[group]
        rows.append({"split": "calibration", "concept": group, "carrier": control["id"],
                     "task": "visible_readout", "mapping": 0, "template": template, "site": "none",
                     "condition": "visible", "strength": 0, "sentence": sentence,
                     "prompt": p["prefix"].format(sentence=sentence) + p["free_templates"][template]})
    random.Random(p["seed"]).shuffle(rows)
    return rows


def build_readout_banks(tokenizer, p):
    banks, audit = {}, {}
    for bank, words in p["word_banks"].items():
        ids, records = set(), []
        for word in words:
            encodings = {text: tokenizer.encode(text, add_special_tokens=False) for text in (word, " " + word)}
            chosen = {e[0] for e in encodings.values() if len(e) == 1}
            if not chosen:
                raise ValueError("No single-token spelling; stop without changing vocabulary: " + word)
            ids.update(chosen)
            records.append({"word": word, "encodings": encodings, "selected_ids": sorted(chosen)})
        banks[bank], audit[bank] = sorted(ids), records
    if set(banks["fruit"]) & set(banks["music"]):
        raise ValueError("Overlapping readout token banks.")
    return banks, audit


def visible_readout_summary(rows, p):
    expected = make_readout_trials(p)
    check_rows(rows, expected)
    scores = {}
    for row in rows:
        a, b = row["fruit_mass"], row["music_mass"]
        if any(not math.isfinite(v) or not 0 < v <= 1 for v in (a, b)) or a + b > 1.000001:
            raise ValueError("Invalid readout probability mass.")
        if row["injection_positions"] != [] or row["actual_norm_ratio"] != 0:
            raise ValueError("Visible readout must not contain an injection.")
        if not math.isclose(row["label_mass"], a + b, abs_tol=1e-6):
            raise ValueError("Inconsistent family probability mass.")
        if not math.isclose(row["p_positive"], a / (a + b), abs_tol=1e-6):
            raise ValueError("Inconsistent conditional probability.")
        scores[(row["carrier"], row["concept"], row["template"])] = math.log(a) - math.log(b)
    groups, gate = {}, p["gate"]
    for concept, template in itertools.product(("fruit", "music"), range(2)):
        rr = [r for r in rows if r["concept"] == concept and r["template"] == template]
        sign = 1 if concept == "fruit" else -1
        odds = [sign * scores[(r["carrier"], concept, template)] for r in rr]
        gains = [sign * (scores[(r["carrier"], concept, template)] - scores[(r["carrier"], "neutral", template)]) for r in rr]
        target = concept + "_mass"
        item = {"count": len(rr), "mean_family_mass": mean(r["label_mass"] for r in rr),
                "mean_target_mass": mean(r[target] for r in rr), "mean_target_log_odds": mean(odds),
                "correct_fraction": mean(v > 0 for v in odds),
                "mean_paired_log_odds_gain": mean(gains), "positive_paired_fraction": mean(v > 0 for v in gains),
                "greedy_target_bank_fraction": mean(r["decision"] == ("positive" if sign == 1 else "negative") for r in rr)}
        item["passes"] = (item["mean_family_mass"] >= gate["min_each_template_mean_family_mass"]
            and item["correct_fraction"] >= gate["min_each_template_correct_fraction"]
            and item["mean_target_log_odds"] > gate["min_each_template_mean_target_log_odds"]
            and item["positive_paired_fraction"] >= gate["min_each_template_positive_paired_fraction"]
            and item["mean_paired_log_odds_gain"] >= gate["min_each_template_mean_paired_log_odds_gain"])
        groups[f"{concept}/template-{template}"] = item
    neutral = {str(t): {"mean_fruit_minus_music_log_odds": mean(scores[(c["id"], "neutral", t)] for c in p["controls"]),
                       "mean_family_mass": mean(r["label_mass"] for r in rows if r["concept"] == "neutral" and r["template"] == t)} for t in range(2)}
    passed = all(g["passes"] for g in groups.values())
    return {"groups": groups, "neutral": neutral, "passes_readout_gates": passed,
            "forward_count": len(rows), "intervention_count": 0, "evaluation_run": False,
            "decision": "Visible readout passed; any hidden-perturbation test requires a separate frozen protocol." if passed else "Stop: visible readout did not pass every frozen gate.",
            "interpretation": "Lexical readout feasibility only; no self-monitoring or conscious-experience inference."}


def execute_readout(p, folder, model, tokenizer, seed):
    # Deliberately has no donor capture, forward-hook registration, or intervention route.
    import torch
    banks, audit = build_readout_banks(tokenizer, p)
    save(folder / "word-bank-tokens.json", {"ids": banks, "spellings": audit})
    trials = make_readout_trials(p)
    save(folder / "calibration-schedule.json", trials)
    rows = []
    with (folder / "calibration.jsonl").open("x", encoding="utf-8") as stream:
        for trial in trials:
            rendered = (tokenizer.bos_token or "") + trial["prompt"]
            token_ids = tokenizer.encode(rendered, add_special_tokens=False)
            seed()
            with torch.no_grad():
                logits = model(torch.tensor([token_ids], device="cuda"), use_cache=False).logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
            a, b = (probabilities[banks[k]].sum().item() for k in ("fruit", "music"))
            top = logits.argmax().item()
            top_ids = torch.topk(logits, 20).indices.tolist()
            row = dict(trial, rendered_prompt=rendered, token_ids=token_ids,
                       injection_positions=[], actual_norm_ratio=0, fruit_mass=a, music_mass=b,
                       label_mass=min(a+b, 1.0), p_positive=a/(a+b), greedy_token_id=top,
                       greedy_token=tokenizer.decode([top]),
                       decision="positive" if top in banks["fruit"] else "negative" if top in banks["music"] else "invalid",
                       bank_probabilities={k: [{"id": i, "text": tokenizer.decode([i]), "p": probabilities[i].item()} for i in ids] for k, ids in banks.items()},
                       top_tokens=[{"id": i, "text": tokenizer.decode([i]), "p": probabilities[i].item()} for i in top_ids])
            stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
            rows.append(row)
            if len(rows) % 8 == 0:
                print(f"Visible readout: {len(rows)}/{len(trials)}", flush=True)
    result = visible_readout_summary(rows, p)
    save(folder / "calibration-summary.json", result)
    save(folder / "completion.json", {"status": "visible_readout_complete", "passes_readout_gates": result["passes_readout_gates"],
         "forward_count": len(rows), "intervention_count": 0, "evaluation_run": False})
    print(json.dumps(result, indent=2), flush=True)


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
                "protocol": p, "protocol_sha256": fingerprint(ROOT / "protocols" / (p["id"] + ".json")),
                "runner_sha256": fingerprint(Path(__file__)),
                "resources": resources, "runtime_review": vet}
    save(folder / "manifest.json", manifest)
    with (folder / "runner-source.py").open("xb") as snapshot:
        snapshot.write(Path(__file__).read_bytes())
    if fingerprint(folder / "runner-source.py") != manifest["runner_sha256"]:
        raise ValueError("Runner snapshot mismatch.")
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
    save(folder / "runtime.json", {"python": sys.version, "torch": torch.__version__,
         "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
         "attention": model.config.text_config._attn_implementation,
         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
    if p["id"] == "pilot-004":
        execute_readout(p, folder, model, tokenizer, seed)
        return
    block = dict(model.named_modules())[p["module"]]
    label_ids = {}
    for label, spellings in p["labels"]["token_spellings"].items():
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in spellings]
        label_ids[label] = sorted({x[0] for x in ids if len(x) == 1})
        if not label_ids[label]:
            raise ValueError("No single-token answer label.")
    if set(label_ids["A"]) & set(label_ids["B"]):
        raise ValueError("Overlapping label token sets.")
    bank_ids = {}
    if p["id"] == "pilot-005":
        bank_ids, audit = build_readout_banks(tokenizer, p)
        save(folder / "word-bank-tokens.json", bank_ids)
        save(folder / "word-bank-spellings.json", audit)
    elif p["id"] in ("pilot-002", "pilot-003"):
        for bank, words in p["word_banks"].items():
            ids = set()
            for word in words:
                encodings = [tokenizer.encode(t, add_special_tokens=False) for t in (word, " " + word)]
                word_ids = {e[0] for e in encodings if len(e) == 1}
                if not word_ids:
                    raise ValueError("Word-bank item has no single-token spelling: " + word)
                ids.update(word_ids)
            bank_ids[bank] = sorted(ids)
        if set(bank_ids["fruit"]) & set(bank_ids["music"]):
            raise ValueError("Overlapping word-bank token sets.")
        save(folder / "word-bank-tokens.json", bank_ids)
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
        rendered = ((tokenizer.bos_token or "") + trial["prompt"]) if trial["task"] == "free_association" else tokenizer.apply_chat_template(
            [{"role": "user", "content": trial["prompt"]}], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        marker = "<sentence>" + trial["sentence"] + "</sentence>"
        if rendered.count(marker) != 1:
            raise ValueError("Carrier marker must occur exactly once.")
        start = rendered.index(marker) + len("<sentence>")
        end = start + len(trial["sentence"])
        positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > start and a < end]
        if not positions or max(positions) >= len(encoded["input_ids"]) - 1:
            raise ValueError("Invalid carrier span.")
        if trial.get("site") == "response":
            positions = [len(encoded["input_ids"]) - 1]
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
            if trial["condition"] in ("sham", "visible"):
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
        if trial["task"] == "free_association":
            concept = next(c for c in p["concepts"] if c["id"] == trial["concept"])
            target_ids, foil_ids = bank_ids[concept["bank"]], bank_ids[concept["foil_bank"]]
            target_mass = probabilities[target_ids].sum().item()
            foil_mass = probabilities[foil_ids].sum().item()
            target_logit = torch.logsumexp(logits[target_ids], dim=0)
            foil_logit = torch.logsumexp(logits[foil_ids], dim=0)
            log_odds = (target_logit - foil_logit).item()
            conditional = torch.softmax(torch.stack([target_logit, foil_logit]), dim=0)[0].item()
            top = logits.argmax().item()
            return dict(trial, token_ids=encoded["input_ids"], rendered_prompt=rendered,
                        injection_positions=positions, actual_norm_ratio=actual_ratios[0] if actual_ratios else 0,
                        target_ids=target_ids, foil_ids=foil_ids, target_mass=target_mass, foil_mass=foil_mass,
                        family_log_odds=log_odds, p_positive=conditional,
                        label_mass=min(target_mass + foil_mass, 1.0), greedy_token_id=top,
                        greedy_token=tokenizer.decode([top]),
                        decision="positive" if top in target_ids else "negative" if top in foil_ids else "invalid")
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
                if p["id"] in ("pilot-003", "pilot-005") and len(rows) == 4:
                    preflight = readout_preflight(rows, p)
                    save(folder / "readout-preflight.json", preflight)
                    print("Visible-control preflight: " + json.dumps(preflight), flush=True)
                    if not preflight["passes"]:
                        break
                if len(rows) % 10 == 0:
                    print(f"{split}: {len(rows)}/{len(trials)} ({time.monotonic()-start:.0f}s)", flush=True)
        if p["id"] == "pilot-005" and len(rows) == len(trials):
            checks = []
            for c, task in itertools.product(p["concepts"], ("free_association", "reading")):
                original = next(r for r in rows if r["condition"] == "sham" and r["concept"] == c["id"]
                                and r["carrier"] == p["carriers"][0]["id"] and r["task"] == task
                                and r["template"] == 0 and r["mapping"] == 0)
                trial = next(t for t in trials if trial_id(t) == trial_id(original))
                repeated = measure(trial)
                checks.append({"original": original, "repeated": repeated, "matches_exactly": original == repeated})
            save(folder / "restoration-checks.json", checks)
            if not all(c["matches_exactly"] for c in checks):
                raise ValueError("Repeated sham changed after temporary interventions; inspect saved checks.")
        return rows
    calibration = calibration_summary(stage("calibration"), p)
    save(folder / "calibration-summary.json", calibration)
    print(json.dumps(calibration, indent=2), flush=True)
    if p["id"] in ("pilot-002", "pilot-003", "pilot-005"):
        save(folder / "completion.json", {"status": "stopped_at_readout_preflight" if calibration.get("preflight", {}).get("passes") is False else "calibration_diagnostic_complete",
             "evaluation_run": False, "selected_by_site": calibration["selected_by_site"]})
        return
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
    parser.add_argument("--protocol", choices=("pilot-001", "pilot-002", "pilot-003", "pilot-004", "pilot-005"))
    args = parser.parse_args()
    manifest = None
    if args.command == "analyze":
        if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
            raise ValueError("A simple run ID is required.")
        folder = ROOT / "results" / args.run_id
        manifest = read(folder / "manifest.json")
        protocol_id = manifest["protocol"]["id"]
        if args.protocol is not None and args.protocol != protocol_id:
            raise ValueError("Requested protocol differs from saved run.")
    else:
        protocol_id = args.protocol or "pilot-005"
    protocol_path = ROOT / "protocols" / (protocol_id + ".json")
    p = validate_protocol(read(protocol_path))
    if args.command == "plan":
        print(json.dumps({"id": p["id"], "protocol_sha256": fingerprint(protocol_path),
             "calibration_trials": len(make_trials(p, "calibration")),
             "evaluation_trials_if_calibration_passes": 0 if p["id"] in ("pilot-002", "pilot-003", "pilot-004", "pilot-005") else len(make_trials(p, "evaluation", p["strengths"][0])),
             "donor_forwards": 0 if p["id"] == "pilot-004" else 12, "repeated_sham_forwards": 4 if p["id"] == "pilot-005" else 0, "status": "Plan only; no ML imports or model access."}, indent=2))
    elif args.command == "run":
        execute(p, args.run_id)
    else:
        if manifest["protocol"] != p or manifest["protocol_sha256"] != fingerprint(protocol_path):
            raise ValueError("Frozen protocol mismatch.")
        rows = [json.loads(line) for line in (folder / "calibration.jsonl").read_text(encoding="utf-8").splitlines()]
        result = calibration_summary(rows, p)
        print(json.dumps(result, indent=2))
        if p["id"] == "pilot-001" and result["selected_strength"] is not None and (folder / "evaluation.jsonl").exists():
            rows = [json.loads(line) for line in (folder / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
            print(json.dumps(evaluation_summary(rows, p, result["selected_strength"]), indent=2))

if __name__ == "__main__":
    main()
