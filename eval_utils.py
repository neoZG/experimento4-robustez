import os, re, torch
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset

def eval_seq_classifier(model, tokenizer, dataset, task_name="NLI", batch_size=32, device="cuda"):
    """Evaluate a seq-classification model on the given dataset.
    Returns list of true labels and list of predicted labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    # We will batch the inference for speed
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i+batch_size]
        texts = []
        if task_name == "NLI":
            # NLI: need to encode premise and hypothesis
            for ex in batch:
                texts.append((ex["premise"], ex["hypothesis"]))
            enc = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True)
        else:
            # SST-2 or other single-sentence classification
            for ex in batch:
                texts.append(ex["sentence"])
            enc = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Move to correct device
        if device == "cuda":
            enc = {k: v.to(torch.device("cuda")) for k,v in enc.items()}
        
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            preds = logits.argmax(dim=-1).cpu().numpy()
        
        # collect results
        labels = [ex["label"] for ex in batch]
        all_preds.extend(preds.tolist())
        all_labels.extend(labels)
    
    return all_labels, all_preds

def parse_answer(output_text, task_name="NLI"):
    """Robust parsing of model answers with regex patterns."""
    output_text = output_text.strip().lower()
    
    # More robust answer extraction with regex
    answer_patterns = [
        r"answer[:\s]*(.+?)(?:\.|$|\n)",
        r"answer is[:\s]*(.+?)(?:\.|$|\n)",
        r"the answer is[:\s]*(.+?)(?:\.|$|\n)",
        r"^(.+?)(?:\.|$|\n)"  # fallback: just first line/sentence
    ]
    
    answer_part = ""
    for pattern in answer_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
            break
    
    if not answer_part:
        answer_part = output_text
    
    # Clean and extract first meaningful word/phrase
    answer_line = re.split(r"[\n\.]", answer_part)[0].strip()
    answer = answer_line.lower()
    
    # Map answer to label with robust pattern matching
    pred_label = None
    if task_name == "NLI":
        # Look for keywords in the answer (case insensitive, partial matching)
        if re.search(r"contradict|contradiction", answer):
            pred_label = 2  # contradiction
        elif re.search(r"entail|entailment", answer):
            pred_label = 0  # entailment
        elif re.search(r"neutral|neither|unknown", answer):
            pred_label = 1  # neutral
    else:  # SST-2
        if re.search(r"positive|good|great|excellent|pos", answer):
            pred_label = 1
        elif re.search(r"negative|bad|terrible|awful|neg", answer):
            pred_label = 0
    
    # Default fallbacks if no pattern matched
    if pred_label is None:
        if task_name == "NLI":
            pred_label = 1  # default to neutral
        else:
            pred_label = 0  # default to negative
    
    return pred_label

def get_generation_config(task_name="NLI"):
    """Get appropriate generation configuration based on task."""
    if task_name == "NLI":
        # NLI needs longer responses: "entailment", "contradiction", "neutral"
        return {"max_new_tokens": 8, "do_sample": False}
    else:  # SST-2
        # Sentiment needs shorter responses: "positive", "negative"
        return {"max_new_tokens": 5, "do_sample": False}

def eval_causal_model(model, tokenizer, dataset, task_name="NLI", device="cuda"):
    """Evaluate a causal LM (generative model) on the given dataset using prompting.
    Returns true labels and predicted labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    # Get task-specific generation configuration
    gen_kwargs = get_generation_config(task_name)
    if hasattr(model.config, 'eos_token_id') and model.config.eos_token_id:
        gen_kwargs["eos_token_id"] = model.config.eos_token_id
    
    for ex in dataset:
        if task_name == "NLI":
            premise = ex["premise"]
            hypothesis = ex["hypothesis"]
            # Construct an instruction prompt for NLI
            prompt = (f"Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: "
                      f"Does the premise entail the hypothesis, contradict it, or is it neutral?\nAnswer:")
        else:  # SST-2 (sentiment)
            sentence = ex["sentence"]
            prompt = f"Sentence: \"{sentence}\"\nQuestion: Is the sentiment of this sentence positive or negative?\nAnswer:"
        
        # Tokenize prompt and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Send inputs to device (for multi-GPU, use cuda:0 for generation input)
        if device == "cuda":
            if hasattr(model, "hf_device_map") and "cuda:0" in str(model.hf_device_map):
                inputs = {k: v.to(torch.device("cuda:0")) for k,v in inputs.items()}
            else:
                inputs = {k: v.to(torch.device("cuda")) for k,v in inputs.items()}
        
        try:
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Use robust answer parsing
            pred_label = parse_answer(output_text, task_name)
            
        except Exception as e:
            print(f"Warning: Generation failed for one example: {e}")
            # Use default fallback
            pred_label = 1 if task_name == "NLI" else 0
        
        all_preds.append(pred_label)
        all_labels.append(ex["label"])
    
    return all_labels, all_preds

def validate_dataset(dataset, dataset_name):
    """Validate that dataset exists and has required fields."""
    if dataset is None or len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_name} is empty or None")
    
    # Check required fields based on dataset type
    sample = dataset[0]
    if "sst" in dataset_name.lower():
        required_fields = ["sentence", "label"]
    else:  # NLI datasets
        required_fields = ["premise", "hypothesis", "label"]
    
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Dataset {dataset_name} missing required field: {field}")
    
    print(f"✓ Dataset {dataset_name} validated: {len(dataset)} examples")
    return True 