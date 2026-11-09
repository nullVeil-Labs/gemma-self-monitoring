"""Existing reviewed runtime guard, preserved from the archived project."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

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
