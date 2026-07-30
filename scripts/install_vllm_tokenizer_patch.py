#!/usr/bin/env python3
"""
Install a .pth file so that the transformers tokenizer patch runs in every Python
process (including vLLM's EngineCore subprocess), fixing:
  AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended

Run once from project root (with the same env you use for trinity):
  python scripts/install_vllm_tokenizer_patch.py

Then (re)start your trinity / vLLM workload.
"""
import os
import site
import sys


def main():
    # Project root: directory containing 'trinity'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    patch_file = os.path.join(
        project_root, "trinity", "common", "models", "transformers_tokenizer_patch.py"
    )
    if not os.path.isfile(patch_file):
        print(f"Patch file not found: {patch_file}", file=sys.stderr)
        sys.exit(1)

    # Use first site-packages (user env)
    site_packages = site.getsitepackages()[0]
    pth_file = os.path.join(site_packages, "trinity_vllm_tokenizer_patch.pth")
    content = f'import runpy; runpy.run_path({repr(patch_file)})\n'

    try:
        with open(pth_file, "w") as f:
            f.write(content)
        print(f"Installed: {pth_file}")
        print("  -> runs transformers tokenizer patch in every Python process (including vLLM EngineCore).")
        print("Restart your trinity / vLLM process for the fix to take effect.")
    except OSError as e:
        print(f"Cannot write {pth_file}: {e}", file=sys.stderr)
        print("Run with a user env or use: sudo python scripts/install_vllm_tokenizer_patch.py", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
