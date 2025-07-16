import random, string, os
import numpy as np
import torch

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For reproducible behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seeds set to {seed} for reproducibility")

def add_noise(text: str) -> str:
    """Add noise (typos, swaps, deletions) to a text."""
    # We'll perform one random character perturbation per text (if length > 4)
    if len(text) < 5:
        return text  # too short to perturb meaningfully
    noisy_text = list(text)
    op = random.choice(["swap", "delete", "insert"])
    idx = random.randrange(len(noisy_text))
    if op == "swap" and idx < len(noisy_text) - 1:
        # swap character with the next one
        noisy_text[idx], noisy_text[idx+1] = noisy_text[idx+1], noisy_text[idx]
    elif op == "delete":
        noisy_text.pop(idx)
    elif op == "insert":
        # insert a random letter near idx
        noisy_text.insert(idx, random.choice(string.ascii_lowercase))
    return "".join(noisy_text)

def subset_dataset(ds, max_count):
    """Reduce dataset size for quick evaluation if configured."""
    if max_count is not None and max_count >= 0:
        max_count = min(max_count, len(ds))
        return ds.select(range(max_count))
    return ds

def validate_environment():
    """Validate that the environment is properly set up."""
    try:
        import transformers
        import datasets
        import sklearn
        import torch
        import yaml
        print("✓ All required packages are available")
        
        # Print versions for reproducibility
        print(f"  - transformers: {transformers.__version__}")
        print(f"  - datasets: {datasets.__version__}")
        print(f"  - torch: {torch.__version__}")
        print(f"  - scikit-learn: {sklearn.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  - CUDA not available, using CPU")
            
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return False

def ensure_output_dir(output_dir="output"):
    """Ensure output directory exists and is writable."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(output_dir, "test_write.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"✓ Output directory '{output_dir}' is ready")
        return True
    except Exception as e:
        print(f"✗ Cannot create or write to output directory '{output_dir}': {e}")
        return False

def validate_config(config_path="config.yaml"):
    """Validate configuration file exists and has required fields."""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ["models", "max_examples", "batch_size", "device"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config missing required field: {field}")
        
        # Validate models configuration
        if not isinstance(config["models"], list) or len(config["models"]) == 0:
            raise ValueError("Config 'models' must be a non-empty list")
        
        for i, model in enumerate(config["models"]):
            if "name" not in model or "type" not in model:
                raise ValueError(f"Model {i} missing 'name' or 'type' field")
        
        print(f"✓ Configuration file '{config_path}' validated")
        return config
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return None 