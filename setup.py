#!/usr/bin/env python3
"""
AI Benchmark Suite Setup Script

Automated setup for BigCode Evaluation Harness with multi-model support.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None, shell=False):
    """Run shell command and handle errors"""
    print(f"Running: {cmd}")
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    result = subprocess.run(cmd, shell=shell, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def setup_bigcode_harness():
    """Set up BigCode Evaluation Harness"""
    base_dir = Path(__file__).parent
    bigcode_dir = base_dir / "bigcode-evaluation-harness"
    
    print("Setting up BigCode Evaluation Harness...")
    
    # Clone if not exists
    if not bigcode_dir.exists():
        if not run_command("git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git", cwd=base_dir):
            return False
    
    # Create virtual environment
    venv_dir = bigcode_dir / "venv"
    if not venv_dir.exists():
        if not run_command([sys.executable, "-m", "venv", "venv"], cwd=bigcode_dir):
            return False
    
    # Get python executable in venv
    if os.name == 'nt':  # Windows
        python_venv = venv_dir / "Scripts" / "python.exe"
        pip_venv = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix/Linux
        python_venv = venv_dir / "bin" / "python"
        pip_venv = venv_dir / "bin" / "pip"
    
    # Install dependencies with venv python
    commands = [
        [str(pip_venv), "install", "--upgrade", "pip"],
        [str(pip_venv), "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"],
        [str(pip_venv), "install", "-r", "requirements.txt"],
        [str(pip_venv), "install", "-e", "."],
    ]
    
    for cmd in commands:
        if not run_command(cmd, cwd=bigcode_dir):
            return False
    
    print("✅ BigCode Evaluation Harness setup complete!")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up AI Benchmark Suite...")
    
    if setup_bigcode_harness():
        print("✅ Setup complete! Ready for benchmarking.")
        print("\nNext steps:")
        print("1. Configure models in config/models.yaml")
        print("2. Set up API keys if using commercial models")
        print("3. Run: python3 run_benchmark.py --help")
    else:
        print("❌ Setup failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()