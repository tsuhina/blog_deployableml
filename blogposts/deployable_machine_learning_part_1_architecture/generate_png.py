#!/usr/bin/env python3
"""Generate PNG image from workflow_diagram.mmd file."""

import subprocess
import sys
from pathlib import Path


def generate_png(mmd_path: Path, output_path: Path, width: int = 2400, height: int = 2880, scale: int = 2):
    """Generate PNG from Mermaid diagram file."""
    mmd_path = Path(mmd_path).resolve()
    output_path = Path(output_path).resolve()
    config_path = mmd_path.parent / "mermaid-config.json"
    
    # Validate files exist
    if not mmd_path.exists():
        print(f"Error: {mmd_path} not found")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Warning: {config_path} not found, proceeding without config")
        config_arg = []
    else:
        config_arg = ["-c", str(config_path)]
    
    # Build mermaid-cli command
    cmd = [
        "npx", "--yes", "@mermaid-js/mermaid-cli@latest",
        "-i", str(mmd_path),
        "-o", str(output_path),
        "-w", str(width),
        "-H", str(height),
        "-b", "white",
        "-s", str(scale)
    ] + config_arg
    
    print(f"Generating PNG from {mmd_path.name}...")
    print(f"  Output: {output_path.name}")
    print(f"  Resolution: {width * scale}x{height * scale} pixels (at {scale}x scale)")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Successfully generated {output_path.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running mermaid-cli: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    mmd_file = script_dir / "workflow_diagram.mmd"
    output_file = script_dir / "workflow_diagram.png"
    
    # High resolution: 2400x2880 with 2x scale = 4800x5760 pixels
    generate_png(mmd_file, output_file, width=2400, height=2880, scale=2)

