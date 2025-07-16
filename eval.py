import os, re, math, yaml, sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
from utils import add_noise, subset_dataset, set_random_seeds, validate_environment, ensure_output_dir, validate_config
from eval_utils import eval_seq_classifier, eval_causal_model, validate_dataset

# Evaluation functions are now imported from eval_utils.py

def main():
    # Set up reproducibility and environment validation
    print("=" * 60)
    print("ROBUSTNESS EVALUATION EXPERIMENT")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please install missing packages.")
        sys.exit(1)
    
    # Validate and load configuration
    config = validate_config("config.yaml")
    if config is None:
        print("Configuration validation failed. Please fix config.yaml.")
        sys.exit(1)
    
    # Ensure output directory exists and is writable
    if not ensure_output_dir("output"):
        print("Cannot create output directory. Please check permissions.")
        sys.exit(1)

    models_config = config["models"]
    max_examples = config["max_examples"]
    batch_size = config.get("batch_size", 16)
    device = config.get("device", "cuda")

    print(f"Configuration loaded: {len(models_config)} models, batch_size={batch_size}, device={device}")

    # Prepare datasets with error handling
    print("\nLoading datasets...")
    try:
        anli = load_dataset("anli")
        sst2 = load_dataset("glue", "sst2")
        xnli = load_dataset("xnli")
        print("✓ All datasets loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        print("Please ensure you have internet connection and datasets can be downloaded.")
        sys.exit(1)

    # Process ANLI test sets
    anli_test_sets = {
        "R1": subset_dataset(anli["test_r1"], max_examples.get("anli", -1)),
        "R2": subset_dataset(anli["test_r2"], max_examples.get("anli", -1)),
        "R3": subset_dataset(anli["test_r3"], max_examples.get("anli", -1))
    }
    # Combined ANLI (concatenate rounds)
    anli_combined = anli_test_sets["R1"].flatten_indices() if hasattr(anli_test_sets["R1"], 'flatten_indices') else anli_test_sets["R1"]
    for rnd in ["R2", "R3"]:
        ds_rnd = anli_test_sets[rnd].flatten_indices() if hasattr(anli_test_sets[rnd], 'flatten_indices') else anli_test_sets[rnd]
        anli_combined = anli_combined.concatenate(ds_rnd)
    anli_test_sets["All"] = anli_combined

    sst2_test = subset_dataset(sst2["test"], max_examples.get("sst2", -1))
    xnli_test = subset_dataset(xnli["test"], max_examples.get("xnli", -1))

    # Filter XNLI to English only
    xnli_en = []
    for ex in xnli_test:
        premise_en = ex["premise"]["en"]
        hypothesis_en = ex["hypothesis"]["en"]
        label = ex["label"]
        xnli_en.append({"premise": premise_en, "hypothesis": hypothesis_en, "label": label})
    xnli_en = Dataset.from_list(xnli_en)

    # Validate all datasets
    try:
        for k, v in anli_test_sets.items():
            validate_dataset(v, f"ANLI-{k}")
        validate_dataset(sst2_test, "SST-2")
        validate_dataset(xnli_en, "XNLI-EN")
    except ValueError as e:
        print(f"✗ Dataset validation failed: {e}")
        sys.exit(1)

    print("\nDatasets loaded and validated. Sizes:")
    for k,v in anli_test_sets.items():
        print(f"  ANLI {k}: {len(v)} examples")
    print(f"  SST-2: {len(sst2_test)} examples")
    print(f"  XNLI (en): {len(xnli_en)} examples")

    # Prepare results storage
    anli_results = []   # list of dict per model with accuracy per round
    robust_results = [] # list of dict per model with SST2/XNLI clean vs noisy metrics
    all_results = {}    # complete results per model

    # Main evaluation loop
    for mconf in models_config:
        model_name = mconf["name"]
        model_type = mconf["type"]
        alias = mconf.get("alias", model_name)
        print(f"\n{'='*50}")
        print(f"Loading model: {alias} ({model_name})")
        print(f"{'='*50}")
        
        # Load tokenizer with error handling
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            print(f"✓ Tokenizer loaded for {alias}")
        except Exception as e:
            print(f"✗ Failed to load tokenizer for {alias}: {e}")
            continue
        
        # Load model with error handling
        try:
            if model_type == "seq_classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else None, trust_remote_code=True)
            else:  # causal LM
                # Use BF16 if available, else FP16 for large models
                dtype = None
                if torch.cuda.is_available():
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
            print(f"✓ Model loaded for {alias}")
        except Exception as e:
            print(f"✗ Failed to load model {alias}: {e}")
            continue

        # Evaluate on tasks
        model_results = {"model": alias}
        
        # Part (a): ANLI zero-shot
        if model_type == "seq_classification" and "mnli" in model_name.lower():
            # Use this model for NLI tasks (ANLI, XNLI)
            print(f"Evaluating {alias} on ANLI (zero-shot)...")
            for round_name, ds in anli_test_sets.items():
                y_true, y_pred = eval_seq_classifier(model, tokenizer, ds, task_name="NLI", batch_size=batch_size, device=device)
                acc = accuracy_score(y_true, y_pred)
                model_results[f"ANLI_{round_name}_acc"] = round(acc * 100, 2)
                print(f"  ANLI-{round_name} Accuracy = {acc*100:.2f}%")
            
            # Part (b): XNLI with noise
            print(f"Evaluating {alias} on XNLI (clean vs noisy)...")
            # Clean XNLI
            y_true_x, y_pred_x = eval_seq_classifier(model, tokenizer, xnli_en, task_name="NLI", batch_size=batch_size, device=device)
            acc_x = accuracy_score(y_true_x, y_pred_x)
            f1_x = f1_score(y_true_x, y_pred_x, average="macro")
            
            # Noisy XNLI
            xnli_noisy = xnli_en.map(lambda ex: {
                "premise": add_noise(ex["premise"]), 
                "hypothesis": add_noise(ex["hypothesis"]), 
                "label": ex["label"]
            })
            y_true_xn, y_pred_xn = eval_seq_classifier(model, tokenizer, xnli_noisy, task_name="NLI", batch_size=batch_size, device=device)
            acc_xn = accuracy_score(y_true_xn, y_pred_xn)
            f1_xn = f1_score(y_true_xn, y_pred_xn, average="macro")
            drop_x = (acc_x - acc_xn) * 100  # drop in percentage points
            
            model_results["XNLI_clean_acc"] = round(acc_x * 100, 2)
            model_results["XNLI_noisy_acc"] = round(acc_xn * 100, 2)
            model_results["XNLI_acc_drop"] = round(drop_x, 2)
            model_results["XNLI_clean_F1"] = round(f1_x * 100, 2)
            model_results["XNLI_noisy_F1"] = round(f1_xn * 100, 2)
            print(f"  XNLI clean vs noisy accuracy: {acc_x*100:.2f}% -> {acc_xn*100:.2f}% (drop {drop_x:.2f} pts)")

        elif model_type == "seq_classification" and "sst-2" in model_name.lower():
            # Use this model for SST-2 tasks
            print(f"Evaluating {alias} on SST-2 (clean vs noisy)...")
            # Clean SST-2
            y_true_s, y_pred_s = eval_seq_classifier(model, tokenizer, sst2_test, task_name="SST2", batch_size=batch_size, device=device)
            acc_s = accuracy_score(y_true_s, y_pred_s)
            f1_s = f1_score(y_true_s, y_pred_s, average="macro")
            
            # Noisy SST-2
            sst2_noisy = sst2_test.map(lambda ex: {"sentence": add_noise(ex["sentence"]), "label": ex["label"]})
            y_true_sn, y_pred_sn = eval_seq_classifier(model, tokenizer, sst2_noisy, task_name="SST2", batch_size=batch_size, device=device)
            acc_sn = accuracy_score(y_true_sn, y_pred_sn)
            f1_sn = f1_score(y_true_sn, y_pred_sn, average="macro")
            drop_s = (acc_s - acc_sn) * 100
            
            model_results["SST2_clean_acc"] = round(acc_s * 100, 2)
            model_results["SST2_noisy_acc"] = round(acc_sn * 100, 2)
            model_results["SST2_acc_drop"] = round(drop_s, 2)
            model_results["SST2_clean_F1"] = round(f1_s * 100, 2)
            model_results["SST2_noisy_F1"] = round(f1_sn * 100, 2)
            print(f"  SST-2 clean vs noisy accuracy: {acc_s*100:.2f}% -> {acc_sn*100:.2f}% (drop {drop_s:.2f} pts)")

        elif model_type == "causal":
            # For causal models, do all tasks via prompting
            print(f"Evaluating {alias} on ANLI (zero-shot, via generation)...")
            # ANLI rounds:
            for round_name, ds in anli_test_sets.items():
                y_true, y_pred = eval_causal_model(model, tokenizer, ds, task_name="NLI", device=device)
                acc = accuracy_score(y_true, y_pred)
                model_results[f"ANLI_{round_name}_acc"] = round(acc * 100, 2)
                print(f"  ANLI-{round_name} Accuracy = {acc*100:.2f}%")
            
            # XNLI:
            print(f"Evaluating {alias} on XNLI (clean vs noisy, via generation)...")
            y_true_x, y_pred_x = eval_causal_model(model, tokenizer, xnli_en, task_name="NLI", device=device)
            acc_x = accuracy_score(y_true_x, y_pred_x)
            f1_x = f1_score(y_true_x, y_pred_x, average="macro")
            
            xnli_noisy_list = []
            for ex in xnli_en:
                xnli_noisy_list.append({
                    "premise": add_noise(ex["premise"]),
                    "hypothesis": add_noise(ex["hypothesis"]),
                    "label": ex["label"]
                })
            xnli_noisy_ds = Dataset.from_list(xnli_noisy_list)
            y_true_xn, y_pred_xn = eval_causal_model(model, tokenizer, xnli_noisy_ds, task_name="NLI", device=device)
            acc_xn = accuracy_score(y_true_xn, y_pred_xn)
            f1_xn = f1_score(y_true_xn, y_pred_xn, average="macro")
            drop_x = (acc_x - acc_xn) * 100
            
            model_results["XNLI_clean_acc"] = round(acc_x * 100, 2)
            model_results["XNLI_noisy_acc"] = round(acc_xn * 100, 2)
            model_results["XNLI_acc_drop"] = round(drop_x, 2)
            model_results["XNLI_clean_F1"] = round(f1_x * 100, 2)
            model_results["XNLI_noisy_F1"] = round(f1_xn * 100, 2)
            print(f"  XNLI clean vs noisy accuracy: {acc_x*100:.2f}% -> {acc_xn*100:.2f}% (drop {drop_x:.2f} pts)")
            
            # SST-2:
            print(f"Evaluating {alias} on SST-2 (clean vs noisy, via generation)...")
            y_true_s, y_pred_s = eval_causal_model(model, tokenizer, sst2_test, task_name="SST2", device=device)
            acc_s = accuracy_score(y_true_s, y_pred_s)
            f1_s = f1_score(y_true_s, y_pred_s, average="macro")
            
            sst2_noisy_list = []
            for ex in sst2_test:
                sst2_noisy_list.append({
                    "sentence": add_noise(ex["sentence"]),
                    "label": ex["label"]
                })
            sst2_noisy_ds = Dataset.from_list(sst2_noisy_list)
            y_true_sn, y_pred_sn = eval_causal_model(model, tokenizer, sst2_noisy_ds, task_name="SST2", device=device)
            acc_sn = accuracy_score(y_true_sn, y_pred_sn)
            f1_sn = f1_score(y_true_sn, y_pred_sn, average="macro")
            drop_s = (acc_s - acc_sn) * 100
            
            model_results["SST2_clean_acc"] = round(acc_s * 100, 2)
            model_results["SST2_noisy_acc"] = round(acc_sn * 100, 2)
            model_results["SST2_acc_drop"] = round(drop_s, 2)
            model_results["SST2_clean_F1"] = round(f1_s * 100, 2)
            model_results["SST2_noisy_F1"] = round(f1_sn * 100, 2)
            print(f"  SST-2 clean vs noisy accuracy: {acc_s*100:.2f}% -> {acc_sn*100:.2f}% (drop {drop_s:.2f} pts)")
        else:
            # If a model doesn't fall into above categories, skip (not expected in this experiment)
            print(f"Skipping model {alias}: unsupported type or usage for this experiment.")
            continue

        # Free up memory before loading next model
        try:
            del model
            del tokenizer
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"✓ Memory cleaned after {alias}")

        # Save results for this model
        anli_results.append({
            "Model": alias,
            "ANLI_R1_acc": model_results.get("ANLI_R1_acc", None),
            "ANLI_R2_acc": model_results.get("ANLI_R2_acc", None),
            "ANLI_R3_acc": model_results.get("ANLI_R3_acc", None),
            "ANLI_All_acc": model_results.get("ANLI_All_acc", None)
        })
        robust_results.append({
            "Model": alias,
            "SST2_clean_acc": model_results.get("SST2_clean_acc", None),
            "SST2_noisy_acc": model_results.get("SST2_noisy_acc", None),
            "SST2_acc_drop": model_results.get("SST2_acc_drop", None),
            "SST2_clean_F1": model_results.get("SST2_clean_F1", None),
            "SST2_noisy_F1": model_results.get("SST2_noisy_F1", None),
            "XNLI_clean_acc": model_results.get("XNLI_clean_acc", None),
            "XNLI_noisy_acc": model_results.get("XNLI_noisy_acc", None),
            "XNLI_acc_drop": model_results.get("XNLI_acc_drop", None),
            "XNLI_clean_F1": model_results.get("XNLI_clean_F1", None),
            "XNLI_noisy_F1": model_results.get("XNLI_noisy_F1", None)
        })
        all_results[alias] = model_results

    # Save results to CSV and JSON with error handling
    try:
        import pandas as pd
        pd.DataFrame(anli_results).to_csv("output/anli_results.csv", index=False)
        pd.DataFrame(robust_results).to_csv("output/robustness_results.csv", index=False)
        
        import json
        with open("output/results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print("✓ Results saved to output/anli_results.csv")
        print("✓ Results saved to output/robustness_results.csv") 
        print("✓ Results saved to output/results.json")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
        print("Results are available in memory but could not be written to files.")

if __name__ == "__main__":
    main() 