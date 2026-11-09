"""Response-only semantic scoring with a frozen benchmark; no intervention route."""
import json
import math
import random


def judge_prompt(text, mapping, p):
    legend = "\n".join(f"{letter} = {category}" for letter, category in zip("ABCD", p["mappings"][mapping]))
    return p["rubric"] + "\n\n" + legend + "\n" + p["classifier_instruction"] + "\nResponse JSON string:\n" + json.dumps(text, ensure_ascii=True)


def benchmark_trials(p):
    rows = [dict(item, mapping=m) for item in p["benchmark"] for m in (0, 1)]
    random.Random(p["seed"]).shuffle(rows)
    return rows


def response_trials(p):
    rows = [{"id": f"{c['id']}-{group}-{t}", "carrier": c["id"], "concept": group, "template": t,
             "sentence": c[group], "prompt": p["prefix"].format(sentence=c[group]) + p["free_templates"][t]}
            for c in p["controls"] for group in ("fruit", "music", "neutral") for t in (0, 1)]
    random.Random(p["seed"] + 1).shuffle(rows)
    return rows


def check_classification(row, text, mapping, p):
    if row["text"] != text or row["mapping"] != mapping or row["judge_prompt"] != judge_prompt(text, mapping, p):
        raise ValueError("Classifier input differs from frozen text-only design.")
    if row["predicted"] not in p["classes"] + ["invalid"]:
        raise ValueError("Unknown semantic label.")
    if not math.isfinite(row["label_mass"]) or not 0 <= row["label_mass"] <= 1:
        raise ValueError("Invalid label probability mass.")
    if set(row["class_probabilities"]) != set(p["classes"]):
        raise ValueError("Incomplete semantic probability vector.")
    probs = row["class_probabilities"].values()
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-6):
        raise ValueError("Invalid semantic probability vector.")


def benchmark_summary(rows, p):
    expected = {(r["id"], r["mapping"]): r for r in benchmark_trials(p)}
    if len(rows) != len(expected) or {(r["id"], r["mapping"]) for r in rows} != set(expected):
        raise ValueError("Missing or duplicate benchmark classifications.")
    for r in rows:
        spec = expected[(r["id"], r["mapping"])]
        check_classification(r, spec["text"], spec["mapping"], p)
        if r["expected"] != spec["expected"] or r["critical"] != spec["critical"]:
            raise ValueError("Benchmark labels changed.")
    accuracy = {label: sum(r["predicted"] == label for r in rows if r["expected"] == label) / sum(r["expected"] == label for r in rows) for label in p["classes"]}
    paired = {item["id"]: [r for r in rows if r["id"] == item["id"]] for item in p["benchmark"]}
    agreement = sum(rr[0]["predicted"] == rr[1]["predicted"] and rr[0]["predicted"] != "invalid" for rr in paired.values()) / len(paired)
    critical = all(r["predicted"] == r["expected"] for r in rows if r["critical"])
    valid = all(r["predicted"] != "invalid" for r in rows)
    gate = p["benchmark_gate"]
    passed = (all(v >= gate["min_each_class_accuracy"] for v in accuracy.values())
        and agreement >= gate["min_mapping_agreement"] and critical and valid
        and min(r["label_mass"] for r in rows) >= gate["min_each_label_mass"])
    return {"rows": len(rows), "accuracy_by_class": accuracy, "mapping_agreement": agreement,
        "all_critical_correct": critical, "all_labels_valid": valid, "minimum_label_mass": min(r["label_mass"] for r in rows),
        "errors": [{"id": r["id"], "text": r["text"], "mapping": r["mapping"], "expected": r["expected"], "predicted": r["predicted"]} for r in rows if r["expected"] != r["predicted"]],
        "passes_benchmark": passed, "self_monitoring_evaluation_run": False}


def response_summary(rows, p):
    expected = {r["id"]: r for r in response_trials(p)}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Missing or duplicate response contexts.")
    cells = []
    for r in rows:
        if any(r[k] != v for k, v in expected[r["id"]].items()):
            raise ValueError("Response context differs from frozen design.")
        judges = r["classifications"]
        if len(judges) != 2 or {j["mapping"] for j in judges} != {0, 1}:
            raise ValueError("Both classifier mappings are required.")
        for j in judges: check_classification(j, r["response_text"], j["mapping"], p)
        agreement = judges[0]["predicted"] == judges[1]["predicted"]
        valid = (r["closed_quote"] and bool(r["response_text"].strip()) and agreement
                 and judges[0]["predicted"] != "invalid"
                 and min(j["label_mass"] for j in judges) >= p["response_gate"]["min_each_label_mass"])
        label = judges[0]["predicted"] if valid else "unscorable"
        target = r["concept"] if r["concept"] != "neutral" else "neither"
        cells.append({"id": r["id"], "concept": r["concept"], "template": r["template"], "response": r["response_text"],
                      "classification": label, "mapping_agreement": agreement, "target_covered": valid and label == target})
    groups = {}
    passed = True
    for category in ("fruit", "music", "neutral"):
        for t in (0, 1):
            rr = [r for r in cells if r["concept"] == category and r["template"] == t]
            coverage = sum(r["target_covered"] for r in rr) / len(rr)
            false_category = sum(r["classification"] in ("fruit", "music", "unscorable") for r in rr) / len(rr)
            ok = coverage >= p["response_gate"]["min_each_category_template_coverage"]
            if category == "neutral": ok &= false_category <= p["response_gate"]["max_each_neutral_template_false_category_rate"]
            groups[f"{category}/template-{t}"] = {"coverage": coverage, "false_category_or_unscorable_rate": false_category, "passes": bool(ok)}
            passed &= ok
    return {"cells": cells, "groups": groups, "passes_response_coverage": bool(passed),
            "interpretation": "Response-only semantic readout feasibility, not self-monitoring or consciousness evidence."}


def execute(p, folder, model, tokenizer, seed, save):
    import torch
    calls = 0
    label_ids = {letter: sorted({ids[0] for text in (letter, " " + letter) if len(ids := tokenizer.encode(text, add_special_tokens=False)) == 1}) for letter in "ABCD"}
    flat = [i for ids in label_ids.values() for i in ids]
    if any(not ids for ids in label_ids.values()) or len(set(flat)) != len(flat):
        raise ValueError("Invalid/overlapping answer letter tokens.")
    save(folder / "classifier-label-tokens.json", label_ids)
    def logits(token_ids):
        nonlocal calls
        if calls >= p["max_forward_calls"]: raise ValueError("Frozen forward-call limit reached.")
        seed()
        with torch.no_grad():
            result = model(torch.tensor([token_ids], device="cuda"), use_cache=False, logits_to_keep=1).logits[0, -1].float()
        calls += 1
        return result
    def classify(text, mapping):
        user_text = judge_prompt(text, mapping, p)
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        values = logits(ids); probabilities = torch.softmax(values, dim=-1)
        masses = {letter: probabilities[ids].sum().item() for letter, ids in label_ids.items()}
        mass = sum(masses.values())
        if mass <= 1e-30: raise ValueError("Classifier answer probabilities underflowed.")
        top = values.argmax().item()
        letter = next((letter for letter, ids in label_ids.items() if top in ids), None)
        return {"text": text, "mapping": mapping, "judge_prompt": user_text, "rendered_prompt": rendered, "token_ids": ids,
            "predicted": p["mappings"][mapping]["ABCD".index(letter)] if letter else "invalid",
            "label_mass": min(mass, 1), "letter_masses": masses,
            "class_probabilities": {category: masses[letter]/mass for letter, category in zip("ABCD", p["mappings"][mapping])},
            "greedy_token_id": top, "greedy_token": tokenizer.decode([top])}
    trials = benchmark_trials(p)
    save(folder / "calibration-schedule.json", trials)
    rows = []
    with (folder / "calibration.jsonl").open("x", encoding="utf-8") as stream:
        for trial in trials:
            r = dict(trial, **{k:v for k,v in classify(trial["text"], trial["mapping"]).items() if k not in trial})
            stream.write(json.dumps(r, allow_nan=False) + "\n"); stream.flush(); rows.append(r)
            if len(rows) % 16 == 0: print(f"Classifier benchmark: {len(rows)}/{len(trials)}", flush=True)
    result = benchmark_summary(rows, p)
    save(folder / "calibration-summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passes_benchmark"]:
        save(folder / "completion.json", {"status": "stopped_at_semantic_benchmark", "response_coverage_run": False, "evaluation_run": False, "forward_calls": calls, "intervention_count": 0})
        return
    trials = response_trials(p)
    save(folder / "responses-schedule.json", trials)
    rows = []
    with (folder / "responses.jsonl").open("x", encoding="utf-8") as stream:
        for trial in trials:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": trial["prompt"]}], tokenize=False, add_generation_prompt=True) + '"'
            prefix = tokenizer.encode(rendered, add_special_tokens=False)
            generated, steps = [], []
            closed = False
            for _ in range(p["max_new_tokens"]):
                values = logits(prefix + generated)
                top = values.argmax().item(); generated.append(top)
                steps.append({"token_id": top, "log_probability": torch.log_softmax(values, dim=-1)[top].item()})
                decoded = tokenizer.decode(generated, skip_special_tokens=False)
                if '"' in decoded: closed = True; break
                if top in {tokenizer.eos_token_id, tokenizer.pad_token_id}: break
            text = decoded.split('"', 1)[0].strip()
            r = dict(trial, rendered_prompt=rendered, prefix_token_ids=prefix, generated_token_ids=generated,
                     generated_text=decoded, response_text=text, generation_steps=steps, closed_quote=closed,
                     classifications=[classify(text, m) for m in (0, 1)])
            stream.write(json.dumps(r, allow_nan=False) + "\n"); stream.flush(); rows.append(r)
            print(f"Fresh response coverage: {len(rows)}/{len(trials)}", flush=True)
    coverage = response_summary(rows, p)
    save(folder / "responses-summary.json", coverage)
    save(folder / "completion.json", {"status": "semantic_readout_complete", "response_coverage_run": True,
         "passes_response_coverage": coverage["passes_response_coverage"], "evaluation_run": False, "forward_calls": calls, "intervention_count": 0})
    print(json.dumps(coverage, indent=2), flush=True)
