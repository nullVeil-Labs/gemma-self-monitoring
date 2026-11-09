"""One entry point for the frozen, objectively graded arithmetic experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arithmetic_runtime", ROOT / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
read = runtime.read


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def validate(p):
    if p["id"] != "arithmetic-001" or len(p["questions"]) != 96:
        raise ValueError("Unexpected protocol.")
    seen = set()
    for q in p["questions"]:
        expected = q["a"] + q["b"] if q["operation"] == "+" else q["a"] * q["b"]
        key = (q["operation"], min(q["a"], q["b"]), max(q["a"], q["b"]))
        if q["expected"] != expected or key in seen or q["expression"] != f"{q['a']} {q['operation']} {q['b']}":
            raise ValueError("Wrong answer key or duplicated problem.")
        seen.add(key)
    for split, n in (("calibration", 8), ("evaluation", 16)):
        for st in p["strata"]:
            if sum(q["split"] == split and q["stratum"] == st["id"] for q in p["questions"]) != n:
                raise ValueError("Unbalanced frozen split.")
    return p


def schedule(p, split):
    rows = [dict(q) for q in p["questions"] if q["split"] == split]
    random.Random(p["seed"] + (split == "evaluation")).shuffle(rows)
    return rows


def prompts(q, mapping, p):
    answer = p["answer_prompt"].format(expression=q["expression"])
    legend = "\n".join(f"{letter} = {meaning}" for letter, meaning in zip("AB", p["label_mappings"][mapping]))
    return p["confidence_prompt"].format(answer_prompt=answer, legend=legend), answer


def grade(text, stopped, expected):
    clean = text.strip()
    valid = bool(stopped and re.fullmatch(r"[+-]?[0-9]+", clean))
    return {"valid_integer": valid, "parsed": int(clean) if valid else None,
            "correct": bool(valid and int(clean) == expected)}


def check_rows(rows, p, split):
    expected = {q["id"]:q for q in schedule(p, split)}
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("Missing or duplicated problem records.")
    for r in rows:
        if any(r[k] != v for k,v in expected[r["id"]].items()): raise ValueError("Frozen question changed.")
        if r["grade"] != grade(r["answer_text"], r["stopped_on_eos"], r["expected"]): raise ValueError("Incorrect grade.")
        if len(r["forecasts"]) != 2 or {f["mapping"] for f in r["forecasts"]} != {0,1}: raise ValueError("Both confidence mappings required.")
        for f in r["forecasts"]:
            if f["prompt"] != prompts(r, f["mapping"], p)[0]: raise ValueError("Confidence input changed.")
            for key in ("p_yes", "label_mass"):
                if not math.isfinite(f[key]) or not 0 <= f[key] <= 1: raise ValueError("Invalid confidence probability.")
            if f["label"] not in ("A", "B", None): raise ValueError("Invalid recorded label.")
        if r["answer_prompt"] != prompts(r, 0, p)[1]: raise ValueError("Answer prompt changed.")


def quality(rows, p):
    correct = sum(r["grade"]["correct"] for r in rows)
    by_mapping = {str(m): {"valid_rate": statistics.mean(f["label"] is not None for r in rows for f in r["forecasts"] if f["mapping"] == m),
                          "mean_label_mass": statistics.mean(f["label_mass"] for r in rows for f in r["forecasts"] if f["mapping"] == m)} for m in (0,1)}
    difference = statistics.mean(abs(r["forecasts"][0]["p_yes"] - r["forecasts"][1]["p_yes"]) for r in rows)
    gate = p["calibration_gate"]
    formats = all(v["valid_rate"] >= gate["min_each_mapping_valid_rate"] and v["mean_label_mass"] >= gate["min_each_mapping_mean_label_mass"] for v in by_mapping.values())
    return {"questions":len(rows), "correct":correct, "incorrect":len(rows)-correct,
            "valid_integer_answers":sum(r["grade"]["valid_integer"] for r in rows), "mapping_quality":by_mapping,
            "mean_mapping_difference":difference, "format_passes":formats and difference <= gate["max_mean_mapping_difference"]}


def calibration_summary(rows, p):
    check_rows(rows,p,"calibration"); q = quality(rows,p); gate=p["calibration_gate"]
    baseline = {st["id"]:(sum(r["grade"]["correct"] for r in rows if r["stratum"]==st["id"])+1)/(sum(r["stratum"]==st["id"] for r in rows)+2) for st in p["strata"]}
    return {**q,"baseline_by_stratum":baseline,"global_baseline":(q["correct"]+1)/(len(rows)+2),
            "passes":q["format_passes"] and q["correct"]>=gate["min_correct"] and q["incorrect"]>=gate["min_incorrect"]}


def confidence(r, mapping=None):
    values = [f["p_yes"] for f in r["forecasts"] if mapping is None or f["mapping"]==mapping]
    return statistics.mean(values)


def within_auc(rows):
    wins, count = 0.0, 0
    for a in rows:
        for b in rows:
            if a["stratum"]==b["stratum"] and a["grade"]["correct"] and not b["grade"]["correct"]:
                pa,pb=confidence(a),confidence(b);wins += 1 if pa>pb else 0.5 if pa==pb else 0;count += 1
    return wins/count if count else None, count


def evaluation_summary(rows, calibration, p):
    check_rows(rows,p,"evaluation"); q=quality(rows,p); baseline=calibration["baseline_by_stratum"]
    def loss(r, prob): return (prob-float(r["grade"]["correct"]))**2
    deltas=[loss(r,baseline[r["stratum"]])-loss(r,confidence(r)) for r in rows]
    grouped={st["id"]:[d for r,d in zip(rows,deltas) if r["stratum"]==st["id"]] for st in p["strata"]}
    rng=random.Random(p["seed"]+3); draws=[]
    for _ in range(p["bootstrap_replicates"]):
        values=[rng.choice(group) for group in grouped.values() for _ in group]
        draws.append(statistics.mean(values))
    draws.sort(); lo=draws[int(0.025*len(draws))]; hi=draws[int(0.975*len(draws))-1]
    auc,pairs=within_auc(rows)
    mapping_gains={str(m):statistics.mean(loss(r,baseline[r["stratum"]])-loss(r,confidence(r,m)) for r in rows) for m in (0,1)}
    improvement=statistics.mean(deltas);gate=p["evaluation_gate"]
    passed=(q["format_passes"] and q["correct"]>=gate["min_correct"] and q["incorrect"]>=gate["min_incorrect"]
            and improvement>=gate["min_brier_improvement"] and lo>0 and auc is not None and auc>=gate["min_within_stratum_auc"] and all(v>0 for v in mapping_gains.values()))
    return {**q,"brier_confidence":statistics.mean(loss(r,confidence(r)) for r in rows),
        "brier_difficulty_baseline":statistics.mean(loss(r,baseline[r["stratum"]]) for r in rows),
        "brier_global_baseline":statistics.mean(loss(r,calibration["global_baseline"]) for r in rows),
        "brier_improvement":improvement,"paired_stratified_bootstrap_95pct":[lo,hi],"mapping_brier_improvements":mapping_gains,
        "within_stratum_auc":auc,"within_stratum_comparable_pairs":pairs,
        "by_stratum":{st["id"]:{"accuracy":statistics.mean(r["grade"]["correct"] for r in rows if r["stratum"]==st["id"]),
            "mean_confidence":statistics.mean(confidence(r) for r in rows if r["stratum"]==st["id"]),"frozen_baseline":baseline[st["id"]]} for st in p["strata"]},
        "passes_pilot_gates":passed,"interpretation":"Candidate held-out performance prediction requiring replication and stronger difficulty controls." if passed else "This pilot did not establish confidence prediction beyond its frozen difficulty-baseline criteria.",
        "consciousness_inference":False}


def execute(p, run_id):
    if not run_id or not re.fullmatch(r"[A-Za-z0-9_-]+",run_id): raise ValueError("Use a unique simple run ID.")
    resources=read(ROOT/"local-resources.json");vet,model_path=runtime.verify_runtime(resources)
    folder=ROOT/"results"/run_id;folder.mkdir(parents=True,exist_ok=False)
    manifest={"run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"protocol":p,"protocol_sha256":digest(ROOT/"protocol.json"),"resources":resources,"runtime_review":vet,"sources":{}}
    for name in ("run.py","runtime.py","tests.py"):
        data=(ROOT/name).read_bytes();manifest["sources"][name]=hashlib.sha256(data).hexdigest()
        with (folder/(name+".snapshot")).open("xb") as f:f.write(data)
    save(folder/"manifest.json",manifest)
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    import torch
    import numpy as np
    from transformers import AutoTokenizer,Gemma3ForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise ValueError("CUDA BF16 required.")
    model=Gemma3ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16).to("cuda").eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    label_ids={letter:sorted({e[0] for text in (letter," "+letter) if len(e:=tokenizer.encode(text,add_special_tokens=False))==1}) for letter in "AB"}
    if any(not ids for ids in label_ids.values()) or set(label_ids["A"]) & set(label_ids["B"]): raise ValueError("Invalid answer-label tokens.")
    stop=model.generation_config.eos_token_id;stop_ids=set(stop if isinstance(stop,list) else [stop])
    stop_ids.add(tokenizer.eos_token_id)
    end=tokenizer.encode("<end_of_turn>",add_special_tokens=False)
    if len(end)==1 and tokenizer.decode(end)=="<end_of_turn>":stop_ids.add(end[0])
    stop_ids.discard(None)
    save(folder/"runtime.json",{"python":sys.version,"torch":torch.__version__,"device":torch.cuda.get_device_name(),"label_ids":label_ids,"stop_ids":sorted(stop_ids),"attention":model.config.text_config._attn_implementation})
    calls=0
    def logits(ids):
        nonlocal calls
        if calls>=p["max_forward_calls"]:raise ValueError("Frozen call bound reached.")
        random.seed(p["seed"]);np.random.seed(p["seed"]);torch.manual_seed(p["seed"]);torch.cuda.manual_seed_all(p["seed"])
        with torch.no_grad():value=model(torch.tensor([ids],device="cuda"),use_cache=False,logits_to_keep=1).logits[0,-1].float()
        calls+=1;return value
    def encode(text):
        rendered=tokenizer.apply_chat_template([{"role":"user","content":text}],tokenize=False,add_generation_prompt=True)
        return rendered,tokenizer.encode(rendered,add_special_tokens=False)
    def measure(q):
        forecasts=[]
        for mapping in (0,1):
            text,answer_prompt=prompts(q,mapping,p);rendered,ids=encode(text);value=logits(ids);probs=torch.softmax(value,dim=-1)
            masses={letter:probs[idx].sum().item() for letter,idx in label_ids.items()};mass=sum(masses.values())
            if mass<1e-30:raise ValueError("Confidence-label probabilities underflowed.")
            top=value.argmax().item();letter=next((l for l,idx in label_ids.items() if top in idx),None)
            yes="AB"[p["label_mappings"][mapping].index("Yes")]
            forecasts.append({"mapping":mapping,"prompt":text,"rendered_prompt":rendered,"token_ids":ids,"p_yes":masses[yes]/mass,
                              "label_mass":min(mass,1),"letter_masses":masses,"label":letter,"greedy_token_id":top,"greedy_token":tokenizer.decode([top])})
        rendered,ids=encode(answer_prompt);generated=[];steps=[];stopped=False
        for _ in range(p["max_answer_tokens"]):
            value=logits(ids+generated);top=value.argmax().item();generated.append(top)
            steps.append({"token_id":top,"log_probability":torch.log_softmax(value,dim=-1)[top].item()})
            if top in stop_ids:stopped=True;break
        answer=tokenizer.decode(generated,skip_special_tokens=True)
        return dict(q,forecasts=forecasts,answer_prompt=answer_prompt,answer_rendered_prompt=rendered,answer_prefix_ids=ids,
                    answer_token_ids=generated,answer_steps=steps,answer_text=answer,stopped_on_eos=stopped,grade=grade(answer,stopped,q["expected"]))
    def stage(split):
        planned=schedule(p,split);save(folder/(split+"-schedule.json"),planned);rows=[]
        with (folder/(split+".jsonl")).open("x",encoding="utf-8") as f:
            for question in planned:
                r=measure(question);f.write(json.dumps(r,allow_nan=False)+"\n");f.flush();rows.append(r)
                if len(rows)%8==0:print(f"{split}: {len(rows)}/{len(planned)}",flush=True)
        return rows
    calibration=calibration_summary(stage("calibration"),p);save(folder/"calibration-summary.json",calibration);print(json.dumps(calibration,indent=2),flush=True)
    if not calibration["passes"]:
        save(folder/"completion.json",{"status":"stopped_at_calibration","evaluation_run":False,"forward_calls":calls,"interventions":0});return
    evaluation=evaluation_summary(stage("evaluation"),calibration,p);save(folder/"evaluation-summary.json",evaluation)
    save(folder/"completion.json",{"status":"complete","evaluation_run":True,"forward_calls":calls,"interventions":0,"passes_pilot_gates":evaluation["passes_pilot_gates"]})
    print(json.dumps(evaluation,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("plan","run","analyze"),nargs="?",default="plan");parser.add_argument("--run-id");args=parser.parse_args()
    p=validate(read(ROOT/"protocol.json"))
    if args.command=="plan":print(json.dumps({"id":p["id"],"protocol_sha256":digest(ROOT/"protocol.json"),"calibration_questions":32,"held_out_questions":64,"maximum_forwards":1728,"model_access":False},indent=2))
    elif args.command=="run":execute(p,args.run_id)
    else:
        if not args.run_id or not re.fullmatch(r"[A-Za-z0-9_-]+",args.run_id):raise ValueError("Run ID required.")
        folder=ROOT/"results"/args.run_id;mf=read(folder/"manifest.json")
        if mf["protocol"]!=p or mf["protocol_sha256"]!=digest(ROOT/"protocol.json"):raise ValueError("Frozen protocol mismatch.")
        rows=[json.loads(l) for l in (folder/"calibration.jsonl").read_text(encoding="utf-8").splitlines()];cal=calibration_summary(rows,p);print(json.dumps(cal,indent=2))
        if (folder/"evaluation.jsonl").exists():
            rows=[json.loads(l) for l in (folder/"evaluation.jsonl").read_text(encoding="utf-8").splitlines()];print(json.dumps(evaluation_summary(rows,cal,p),indent=2))


if __name__=="__main__":main()
