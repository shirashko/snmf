import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from evaluation.utils.validation_functions import (
    get_arithmetic_eval_fn,
    get_both_wmdp_eval_fn,
    get_wmdp_bio_eval_fn,
    get_wmdp_cyber_eval_fn,
)
import argparse
import json
import os

def resolve_eval_device(requested_device: str) -> str:
    """Choose a safe device for this environment."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "auto":
        requested_device = "cuda"

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[eveluate_model.py] CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        major, minor = torch.cuda.get_device_capability(0)
        # This environment's installed PyTorch wheels require sm_70+.
        if major < 7:
            name = torch.cuda.get_device_name(0)
            print(
                f"[eveluate_model.py] GPU {name} has compute capability sm_{major}{minor}, "
                "which is unsupported by this PyTorch build. Falling back to CPU."
            )
            return "cpu"
        return requested_device

    return requested_device

def run_standalone_eval(
    model_path,
    eval_mode="arithmetic",
    large_eval=False,
    no_mmlu=False,
    device="cuda",
    batch_size=16,
    max_length=256,
    cache_dir="./cache",
    dataset_cache_dir="./cache",
    eng_valid_file="data/valid_eng.jsonl",
):
    resolved_device = resolve_eval_device(device)
    # When we load the model on CPU (e.g. sm_61 GPU + PyTorch sm_70+ wheels), the default
    # Accelerator still prepares dataloaders for CUDA if a GPU is visible, causing
    # "index on cuda, weights on cpu" in embedding. Force CPU placement when resolved_device is cpu.
    accelerator = Accelerator(cpu=(resolved_device == "cpu"))
    dtype = torch.bfloat16 if resolved_device != "cpu" else torch.float32
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=dtype,
        device_map=resolved_device
    )

    if eval_mode == "arithmetic":
        # Configuration for arithmetic evaluation
        # Note: Ensure paths to validation files exist
        eval_factory = get_arithmetic_eval_fn(
            model_name=model_path,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            dataset_cache_dir=dataset_cache_dir,
            num_wiki_batches=50,
            eng_valid_file=eng_valid_file,  # Required for CE loss check
            accelerator=accelerator
        )
    elif eval_mode == "wmdp_bio":
        eval_factory = get_wmdp_bio_eval_fn(
            accelerator=accelerator,
            large_eval=large_eval,
            no_mmlu=no_mmlu,
        )
    elif eval_mode == "wmdp_cyber":
        eval_factory = get_wmdp_cyber_eval_fn(
            accelerator=accelerator,
            large_eval=large_eval,
            no_mmlu=no_mmlu,
        )
    elif eval_mode == "both_wmdp":
        if no_mmlu:
            print("[eveluate_model.py] --no-mmlu is ignored for eval-mode=both_wmdp.")
        eval_factory = get_both_wmdp_eval_fn(
            accelerator=accelerator,
            large_eval=large_eval,
        )
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")
    
    # Run the evaluation
    # This returns a dict with accuracy for each operation (e.g., 'val/addition_acc')
    results = eval_factory(model, print_results=True)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone arithmetic or WMDP evaluation.")
    parser.add_argument("--model-path", default="local_models/gemma-2-0.3B_reference_model")
    parser.add_argument(
        "--eval-mode",
        default="arithmetic",
        choices=["arithmetic", "wmdp_bio", "wmdp_cyber", "both_wmdp"],
        help="Which evaluation pipeline to run.",
    )
    parser.add_argument(
        "--large-eval",
        action="store_true",
        help="Use larger/full evaluation limits for WMDP/MMLU tasks.",
    )
    parser.add_argument(
        "--no-mmlu",
        action="store_true",
        help="For single-domain WMDP modes, skip MMLU.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--dataset-cache-dir", default="./cache")
    parser.add_argument("--eng-valid-file", default="data/valid_eng.jsonl")
    parser.add_argument("--output-json", default=None, help="Optional path to write eval results as JSON.")
    args = parser.parse_args()

    model_results = run_standalone_eval(
        model_path=args.model_path,
        eval_mode=args.eval_mode,
        large_eval=args.large_eval,
        no_mmlu=args.no_mmlu,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        eng_valid_file=args.eng_valid_file,
    )
    print(model_results)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2, sort_keys=True)
        print(f"[eveluate_model.py] Wrote results to {args.output_json}")