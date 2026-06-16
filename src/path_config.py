"""
Path Configuration for T-Few Project

This file centralizes all filesystem paths that may differ between environments
(Windows, Linux, macOS). Update the ROOT_DIR constant below to match your system.

Examples:
  For macOS:   ROOT_DIR = "/Users/work"
  For Linux:   ROOT_DIR = "/work"
  For Windows: ROOT_DIR = "C:/work" or "D:/work"
"""

import os

# ============================================================================
# CONFIGURE YOUR ROOT DIRECTORY HERE
# ============================================================================

# Set ROOT_DIR to match your environment:
ROOT_DIR = "/Users/work"  # <-- EDIT THIS LINE for your system

# ============================================================================
# DERIVED PATHS (Don't modify these unless you have custom paths)
# ============================================================================

# T-Few project root (parent of src/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# Experiment output folder (stores per-run results such as dev_scores.json)
EXP_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "exp_out")

# Summary output folder (stores aggregated CSV result tables)
SUMMARY_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "exp_out")

# TabLLM related paths
TABLLM_ROOT = os.path.join(ROOT_DIR, "TabLLM")
DATASETS_OFFLINE = os.path.join(TABLLM_ROOT, "datasets_serialized")
TEMPLATES_DIR = os.path.join(TABLLM_ROOT, "templates")
ICO_CONFIG_PATH = os.path.join(TABLLM_ROOT, "ico_config.py")

# Helper function to get template file path
def get_template_path(task_name):
    """Get the path to a template YAML file for a given task"""
    return os.path.join(TEMPLATES_DIR, f"templates_{task_name}.yaml")

# ============================================================================
# VALIDATION (Optional - checks if paths exist)
# ============================================================================

def validate_paths(verbose=False):
    """
    Validate that configured paths exist.
    Returns True if all paths exist, False otherwise.
    """
    paths_to_check = {
        "ROOT_DIR": ROOT_DIR,
        "EXP_OUTPUT_PATH": EXP_OUTPUT_PATH,
        "SUMMARY_OUTPUT_PATH": SUMMARY_OUTPUT_PATH,
        "TABLLM_ROOT": TABLLM_ROOT,
        "DATASETS_OFFLINE": DATASETS_OFFLINE,
        "TEMPLATES_DIR": TEMPLATES_DIR,
    }
    
    all_exist = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        if verbose:
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist

# Print warning if paths don't exist (only on first import)
if __name__ != "__main__":
    if not validate_paths(verbose=False):
        import warnings
        warnings.warn(
            f"Some configured paths in src/path_config.py don't exist. "
            f"ROOT_DIR is set to: {ROOT_DIR}. "
            f"Please update ROOT_DIR in src/path_config.py to match your environment.",
            UserWarning
        )

# For debugging - run this file directly to check paths
if __name__ == "__main__":
    print("=" * 70)
    print("T-Few Path Configuration")
    print("=" * 70)
    print(f"\nConfigured ROOT_DIR: {ROOT_DIR}")
    print("\nPath Validation:")
    validate_paths(verbose=True)
    print("\n" + "=" * 70)
    print("\nIf paths are incorrect, edit ROOT_DIR in src/path_config.py")
    print("=" * 70)
