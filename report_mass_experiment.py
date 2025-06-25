import os
import glob
import json
from collections import defaultdict
import argparse

def load_results_from_dir(results_dir, model_name):
    results = []
    for file_path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Accept both 'llm' and 'model' keys for compatibility
            model_key = data.get("llm") or data.get("model")
            if model_key == model_name:
                results.append(data)
    return results

def summarize_results(results):
    summary = defaultdict(list)
    for result in results:
        # Collect top-level stats if present
        for key in ['sample_count', 'legal_accuracy', 'next_state_accuracy', 'legal_next_state_accuracy', 'legal_count']:
            if key in result:
                summary[key].append(result[key])
        # Optionally, aggregate per-sample stats if needed
        if 'samples' in result:
            for sample in result['samples']:
                for k, v in sample.items():
                    summary[f"sample_{k}"].append(v)
    return summary

def print_summary(summary, model_name):
    print(f"Summary for model: {model_name}")
    # Print total sample count
    if "sample_count" in summary:
        total_samples = sum(summary["sample_count"])
        print(f"Total samples: {total_samples}")
    # Print total legal count
    if "legal_count" in summary:
        total_legal = sum(summary["legal_count"])
        print(f"Total legal: {total_legal}")
    # Print accuracy rates
    if "legal_accuracy" in summary and "sample_count" in summary:
        total_legal_acc = sum(summary["legal_accuracy"])
        total_samples = sum(summary["sample_count"])
        print(f"Legal accuracy: {total_legal_acc}/{total_samples} = {total_legal_acc/total_samples:.3f}")
    if "next_state_accuracy" in summary and "sample_count" in summary:
        total_next_state_acc = sum(summary["next_state_accuracy"])
        total_samples = sum(summary["sample_count"])
        print(f"Next state accuracy: {total_next_state_acc}/{total_samples} = {total_next_state_acc/total_samples:.3f}")
    if "legal_next_state_accuracy" in summary and "legal_count" in summary:
        total_legal_next_state_acc = sum(summary["legal_next_state_accuracy"])
        total_legal = sum(summary["legal_count"])
        if total_legal > 0:
            print(f"Legal next state accuracy: {total_legal_next_state_acc}/{total_legal} = {total_legal_next_state_acc/total_legal:.3f}")
        else:
            print("Legal next state accuracy: N/A (no legal samples)")
    # Print per-file stats for reference, but skip individual sample keys
    print("\nPer-file stats (averages):")
    for key, values in summary.items():
        if key.startswith("sample_"):
            continue
        if all(isinstance(v, (int, float)) for v in values):
            avg = sum(values) / len(values)
            print(f"{key}: avg={avg:.3f}, min={min(values)}, max={max(values)}, count={len(values)}")
        else:
            print(f"{key}: {values[:5]}{'...' if len(values) > 5 else ''}")

def print_summary_by_game(summary, results, model_name):
    print(f"Summary for model: {model_name} (by game)")
    # Collect all game names from the results
    game_names = set()
    for result in results:
        if "dataset" in result:
            # Try to extract game name from dataset filename
            base = os.path.basename(result["dataset"])
            game_name = base.split("_")[0]
            game_names.add(game_name)
        elif "game_name" in result:
            game_names.add(result["game_name"])
    if not game_names:
        print("No game names found in results.")
        return

    for game in sorted(game_names):
        # Filter results for this game
        game_results = []
        for result in results:
            if "dataset" in result:
                base = os.path.basename(result["dataset"])
                game_name = base.split("_")[0]
                if game_name == game:
                    game_results.append(result)
            elif "game_name" in result and result["game_name"] == game:
                game_results.append(result)
        if not game_results:
            continue
        # Summarize for this game
        game_summary = summarize_results(game_results)
        print(f"\n=== {game} ===")
        # Print total sample count
        if "sample_count" in game_summary:
            total_samples = sum(game_summary["sample_count"])
            print(f"Total samples: {total_samples}")
        if "legal_count" in game_summary:
            total_legal = sum(game_summary["legal_count"])
            print(f"Total legal: {total_legal}")
        if "legal_accuracy" in game_summary and "sample_count" in game_summary:
            total_legal_acc = sum(game_summary["legal_accuracy"])
            total_samples = sum(game_summary["sample_count"])
            print(f"Legal accuracy: {total_legal_acc}/{total_samples} = {total_legal_acc/total_samples:.3f}")
        if "next_state_accuracy" in game_summary and "sample_count" in game_summary:
            total_next_state_acc = sum(game_summary["next_state_accuracy"])
            total_samples = sum(game_summary["sample_count"])
            print(f"Next state accuracy: {total_next_state_acc}/{total_samples} = {total_next_state_acc/total_samples:.3f}")
        if "legal_next_state_accuracy" in game_summary and "legal_count" in game_summary:
            total_legal_next_state_acc = sum(game_summary["legal_next_state_accuracy"])
            total_legal = sum(game_summary["legal_count"])
            if total_legal > 0:
                print(f"Legal next state accuracy: {total_legal_next_state_acc}/{total_legal} = {total_legal_next_state_acc/total_legal:.3f}")
            else:
                print("Legal next state accuracy: N/A (no legal samples)")
        print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Directory with experiment result JSON files")
    args = parser.parse_args()

    # Find all models in the results directory
    model_names = set()
    for file_path in glob.glob(os.path.join(args.results_dir, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            model_key = data.get("llm") or data.get("model")
            if model_key:
                model_names.add(model_key)

    if not model_names:
        print(f"No models found in {args.results_dir}")
        return

    model_names = sorted(model_names)
    print("Available models:")
    for idx, model in enumerate(model_names, 1):
        print(f"{idx}. {model}")

    while True:
        try:
            selection = int(input("Select a model by number: "))
            if 1 <= selection <= len(model_names):
                selected_model = model_names[selection - 1]
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    results = load_results_from_dir(args.results_dir, selected_model)
    if not results:
        print(f"No results found for model '{selected_model}' in {args.results_dir}")
        return

    summary = summarize_results(results)
    print_summary(summary, selected_model)
    print_summary_by_game(summary, results, selected_model)

if __name__ == "__main__":
    main()