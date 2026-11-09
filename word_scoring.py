"""Complete quoted-word likelihoods; no activation intervention or model writes."""
import math


def logsumexp(values):
    values = list(values)
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Expected finite nonempty log probabilities.")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def candidate_prefixes(candidates):
    sequences = [tuple(c["token_ids"]) for c in candidates]
    if not sequences or any(not seq for seq in sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("Candidate token paths must be nonempty and unique.")
    ordered = sorted(sequences)
    if any(b[:len(a)] == a for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Candidate events overlap: one complete path prefixes another.")
    nodes = {}
    for seq in sequences:
        for i, token in enumerate(seq):
            nodes.setdefault(seq[:i], set()).add(token)
    return {node: sorted(children) for node, children in nodes.items()}


def make_candidates(tokenizer, rendered, p):
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    records = {}
    for bank, words in p["word_banks"].items():
        for word in words:
            for variant in (word, word.capitalize(), word.upper()):
                for leading in ("", " "):
                    text = leading + variant + '"'
                    complete = tokenizer.encode(rendered + text, add_special_tokens=False)
                    if complete[:len(prefix)] != prefix:
                        raise ValueError("Candidate changes prefix tokenization; no approximation permitted.")
                    suffix = tuple(complete[len(prefix):])
                    if not 1 <= len(suffix) <= p["max_suffix_tokens"]:
                        raise ValueError("Candidate suffix exceeds frozen token bound.")
                    if suffix in records:
                        if records[suffix]["bank"] != bank:
                            raise ValueError("Candidate banks overlap.")
                        records[suffix]["spellings"].append(text)
                    else:
                        records[suffix] = {"bank": bank, "word": word, "spellings": [text], "token_ids": list(suffix)}
    candidates = list(records.values())
    nodes = candidate_prefixes(candidates)
    if len(nodes) > p["max_prefix_nodes_per_context"]:
        raise ValueError("Candidate prefix count exceeds frozen bound.")
    return prefix, candidates, nodes


def score_paths(candidates, edges):
    records = []
    for c in candidates:
        tokens = c["token_ids"]
        logs = [edges[tuple(tokens[:i])][token] for i, token in enumerate(tokens)]
        if any(not math.isfinite(v) or v > 1e-6 for v in logs):
            raise ValueError("Invalid conditional log probability.")
        records.append(dict(c, token_log_probabilities=logs, log_probability=sum(logs)))
    bank_logs = {bank: logsumexp(r["log_probability"] for r in records if r["bank"] == bank) for bank in ("fruit", "music")}
    total_log = logsumexp(bank_logs.values())
    if total_log > 1e-5:
        raise ValueError("Probability of disjoint candidate events exceeds one.")
    return records, bank_logs


def summarize(rows, p):
    by_key = {(r["carrier"], r["concept"], r["template"]): r for r in rows}
    expected = {(c["id"], group, t) for c in p["controls"] for group in ("fruit", "music", "neutral") for t in (0, 1)}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("Missing or duplicate complete-word control cells.")
    for r in rows:
        if r["injection_positions"] or r["actual_norm_ratio"] != 0:
            raise ValueError("Complete-word controls must not contain interventions.")
        candidates = r["candidates"]
        candidate_prefixes(candidates)
        if {c["word"] for c in candidates} != {w for words in p["word_banks"].values() for w in words}:
            raise ValueError("Candidate vocabulary differs from protocol.")
        for c in candidates:
            if c["word"] not in p["word_banks"][c["bank"]] or len(c["token_ids"]) != len(c["token_log_probabilities"]):
                raise ValueError("Invalid candidate bank or token alignment.")
            if any(not math.isfinite(v) or v > 1e-6 for v in c["token_log_probabilities"]):
                raise ValueError("Invalid candidate token score.")
            if not math.isclose(c["log_probability"], sum(c["token_log_probabilities"]), abs_tol=1e-10):
                raise ValueError("Candidate score differs from token sum.")
        bank_logs = {bank: logsumexp(c["log_probability"] for c in candidates if c["bank"] == bank) for bank in ("fruit", "music")}
        for bank, value in bank_logs.items():
            if not math.isclose(value, r[bank + "_log_mass"], abs_tol=1e-10):
                raise ValueError("Bank score differs from complete-word probabilities.")
        if logsumexp(bank_logs.values()) > 1e-5:
            raise ValueError("Invalid total candidate probability.")
    cells = []
    gate = p["gate"]
    for c in p["controls"]:
        for concept in ("fruit", "music"):
            for template in (0, 1):
                r = by_key[(c["id"], concept, template)]
                neutral = by_key[(c["id"], "neutral", template)]
                sign = 1 if concept == "fruit" else -1
                score = sign * (r["fruit_log_mass"] - r["music_log_mass"])
                delta = score - sign * (neutral["fruit_log_mass"] - neutral["music_log_mass"])
                mass = math.exp(logsumexp((r["fruit_log_mass"], r["music_log_mass"])))
                failed = []
                if mass < gate["min_each_cell_family_mass"]: failed.append("family_mass")
                if score <= gate["min_each_cell_target_log_odds"]: failed.append("category_preference")
                if delta < gate["min_each_cell_paired_log_odds_gain"]: failed.append("paired_neutral_gain")
                cells.append({"carrier": c["id"], "concept": concept, "template": template,
                    "family_mass": mass, "target_log_odds": score, "paired_log_odds_gain": delta,
                    "passes": not failed, "failed_gates": failed})
    return {"cells": cells, "passing_cells": sum(c["passes"] for c in cells), "total_category_cells": len(cells),
            "passes_readout_gates": all(c["passes"] for c in cells), "contexts": len(rows),
            "model_forward_calls": sum(r["forward_calls"] for r in rows),
            "prefix_sequences_scored": sum(r["prefix_count"] for r in rows),
            "serial_check_max_log_probability_difference": max(r["serial_check_max_difference"] for r in rows),
            "evaluation_run": False, "intervention_count": 0,
            "interpretation": "Visible complete-word likelihood validation only; no self-monitoring or consciousness inference."}


def evaluate(model, tokenizer, p, trials, seed):
    import torch
    total_calls = 0
    for trial in trials:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": trial["prompt"]}],
                      tokenize=False, add_generation_prompt=True) + '"'
        prefix, candidates, nodes = make_candidates(tokenizer, rendered, p)
        edges, forward_calls, serial_difference = {}, 0, 0.0
        serial_checked = False
        top_tokens = []
        def forward(paths):
            nonlocal total_calls, forward_calls
            if total_calls >= p["max_forward_calls"]:
                raise ValueError("Frozen forward-call limit reached.")
            seed()
            ids = torch.tensor([prefix + list(path) for path in paths], device="cuda")
            with torch.no_grad():
                logits = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                values = torch.log_softmax(logits, dim=-1)
            total_calls += 1
            forward_calls += 1
            return values
        for depth in sorted({len(path) for path in nodes}):
            paths = sorted(path for path in nodes if len(path) == depth)
            for start in range(0, len(paths), p["batch_size"]):
                batch = paths[start:start + p["batch_size"]]
                values = forward(batch)
                if depth == 0:
                    top_ids = torch.topk(values[0], 20).indices.tolist()
                    top_tokens = [{"id": i, "text": tokenizer.decode([i]), "log_probability": values[0, i].item()} for i in top_ids]
                if not serial_checked and (len(batch) > 1 or (p["batch_size"] == 1 and depth > 0)):
                    serial = forward([batch[0]])
                    ids = nodes[batch[0]]
                    serial_difference = (serial[0, ids] - values[0, ids]).abs().max().item()
                    if serial_difference > p["max_serial_log_probability_difference"]:
                        raise ValueError(f"Numerical repeatability discrepancy {serial_difference:.9g} exceeds frozen tolerance.")
                    serial_checked = True
                for i, path in enumerate(batch):
                    edges[path] = {token: values[i, token].item() for token in nodes[path]}
        if not serial_checked:
            raise ValueError("Numerical repeatability check was not exercised.")
        records, logs = score_paths(candidates, edges)
        total_log = logsumexp(logs.values())
        yield dict(trial, rendered_prompt=rendered, token_ids=prefix,
            injection_positions=[], actual_norm_ratio=0, candidates=records,
            fruit_log_mass=logs["fruit"], music_log_mass=logs["music"], label_mass=math.exp(total_log),
            p_positive=math.exp(logs["fruit"] - total_log), decision="invalid", top_next_tokens=top_tokens,
            prefix_count=len(nodes), forward_calls=forward_calls, serial_check_max_difference=serial_difference)
