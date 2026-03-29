import argparse
import json
import torch
from pathlib import Path
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
torch.serialization.add_safe_globals([GemmaTokenizerFast])

from utils import resolve_device, set_seed
from model_utils import load_local_model
from supervised_analysis import analyze_features_supervised, plot_layer_concept_trends
from unsupervised_analysis import analyze_features_unsupervised


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-trained SNMF results.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True, help="Path to folder containing layer_X subfolders")
    parser.add_argument(
        "--role-assignment-threshold",
        type=float,
        default=0.15,
        metavar="LOG_RATIO",
        help="Minimum log-ratio (natural log) to assign a strong role_label vs weak_mixed / neutral_lean.",
    )
    parser.add_argument("--skip-vocab", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--top-k-unsupervised", type=int, default=30)
    parser.add_argument("--top-k-supervised", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    for layer_folder in sorted(results_dir.glob("layer_*")):
        layer_num = int(layer_folder.name.split("_")[1])
        factors_path = layer_folder / "snmf_factors.pt"

        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because it doesn't exist.")
            continue

        output_json_path = layer_folder / "feature_analysis_supervised.json"

        print(f"\nProcessing {layer_folder.name}...")
        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)
        F, G = checkpoint['F'], checkpoint['G']
        token_ids, sample_ids = checkpoint['token_ids'], checkpoint['sample_ids']
        labels, mode = checkpoint['labels'], checkpoint.get('mode', 'mlp_intermediate')

        # 1. Base statistical profiling
        supervised_results = analyze_features_supervised(
            G,
            labels,
            sample_ids,
            token_ids,
            local_model.tokenizer,
            save_raw=args.save_raw,
            top_k=args.top_k_supervised,
            role_assignment_threshold=args.role_assignment_threshold,
        )

        with open(output_json_path, 'w') as f:
            json.dump(supervised_results, f, indent=2)

        # Unsupervised Vocabulary Projection (Logit Lens)
        if not args.skip_vocab:
            unsupervised_results = analyze_features_unsupervised(
                F=F,
                local_model=local_model,
                layer=layer_num,
                mode=mode,
                top_k_tokens=args.top_k_unsupervised
            )
            with open(layer_folder / "feature_analysis_unsupervised.json", 'w') as f:
                json.dump(unsupervised_results, f, indent=2)

    print("\nGenerating model-wide trend plots...")
    try:
        plot_layer_concept_trends(args.results_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print(f"\nAnalysis complete. Files saved in {args.results_dir}")


if __name__ == "__main__":
    main()


"""
python analyze_snmf_results.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --results-dir "./final_run_all_layers" \
    --role-assignment-threshold 0.15 \
    --top-k-supervised 50 \
    --top-k-unsupervised 64 \
    --save-raw
    
    
python analyze_snmf_results.py \
    --model-path "models/gemma2-2.03B_pretrained" \
    --results-dir "./pretrained_results" \
    --role-assignment-threshold 0.15 \
    --top-k-supervised 50 \
    --top-k-unsupervised 64 \
    --save-raw
"""