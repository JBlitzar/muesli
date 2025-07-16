#!/usr/bin/env python3
"""
Compile Qt resources file to Python module.

This script compiles the resources.qrc file into a Python module (resources_rc.py)
using the PySide6 rcc tool. The resulting module can be imported to access
resources in the application.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #

def log(msg: str, verbose: bool = False) -> None:  # Lightweight logger
    if verbose:
        print(msg)

def find_rcc_tool():
    """Find the PySide6 rcc tool in the system."""
    try:
        # Try to import PySide6 to get its location
        import PySide6
        pyside_dir = Path(PySide6.__file__).parent
        
        # Check platform-specific executable name
        if sys.platform == "win32":
            rcc_tool = pyside_dir / "rcc.exe"
        else:
            rcc_tool = pyside_dir / "rcc"
        
        # If direct path doesn't exist, try in the bin directory
        if not rcc_tool.exists():
            if sys.platform == "win32":
                rcc_tool = pyside_dir / "bin" / "rcc.exe"
            else:
                rcc_tool = pyside_dir / "bin" / "rcc"
        
        if rcc_tool.exists():
            return str(rcc_tool)
        
        # If still not found, try to find it in PATH
        try:
            result = subprocess.run(
                ["which", "rcc" if sys.platform != "win32" else "rcc.exe"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Last resort: try pyside6-rcc which is often in PATH
        try:
            result = subprocess.run(
                ["which", "pyside6-rcc"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
            
    except ImportError:
        print("PySide6 not installed. Please install it with: pip install PySide6")
        sys.exit(1)
    
    print("Could not find PySide6 rcc tool. Make sure PySide6 is installed correctly.")
    sys.exit(1)

def ensure_qrc_file(script_dir: Path, *, verbose: bool = False) -> Path:
    """
    Verify that ``resources.qrc`` exists and that it contains the correct
    reference to ``resources/loading.svg``.  If the file is missing or the
    path entry is wrong, (re)generate a minimal QRC file.
    """
    qrc_file = script_dir / "resources.qrc"
    resources_dir = script_dir / "resources"
    svg_path = resources_dir / "loading.svg"

    # Create resources directory if it does not exist
    resources_dir.mkdir(parents=True, exist_ok=True)

    # (Re-)generate QRC file if it is missing or incorrect
    regenerate = not qrc_file.exists()
    if not regenerate:
        try:
            tree = ET.parse(qrc_file)
            root = tree.getroot()
            found = any(elem.text == "resources/loading.svg" for elem in root.iter("file"))
            regenerate = not found
        except ET.ParseError:
            regenerate = True

    if regenerate:
        log(f"[compile_resources] Generating {qrc_file}", verbose)
        qrc_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<RCC>\n"
            '    <qresource prefix="/images/resources">\n'
            "        <file>resources/loading.svg</file>\n"
            "    </qresource>\n"
            "</RCC>\n"
        )
        qrc_file.write_text(qrc_content, encoding="utf-8")
    else:
        log(f"[compile_resources] Found existing {qrc_file}", verbose)

    # Basic sanity check for the SVG itself
    if not svg_path.exists():
        print(f"Warning: SVG file not found at {svg_path}. UI may miss loader icon.")

    return qrc_file

def compile_resources():
    """Compile the resources.qrc file to a Python module."""
    parser = argparse.ArgumentParser(
        description="Compile Qt .qrc to Python module via PySide6 rcc."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Ensure QRC file is present and correct
    qrc_file = ensure_qrc_file(script_dir, verbose=args.verbose)
    
    # Path for the output Python module
    output_file = script_dir / "resources_rc.py"
    
    if not qrc_file.exists():
        print(f"Error: Resource file not found: {qrc_file}")
        sys.exit(1)
    
    # Find the rcc tool
    rcc_tool = find_rcc_tool()
    
    log(f"[compile_resources] Using rcc tool: {rcc_tool}", args.verbose)
    print(f"Compiling {qrc_file} -> {output_file}")
    
    try:
        # Run the rcc tool
        result = subprocess.run(
            [rcc_tool, "-g", "python", str(qrc_file), "-o", str(output_file)],
            check=True,
            capture_output=True,
            text=True
        )
        if args.verbose and result.stdout:
            print(result.stdout.strip())
        print(f"Successfully compiled resources to {output_file}")
    except subprocess.CalledProcessError as e:
        if args.verbose and e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        print(f"Error compiling resources: {e}")
        sys.exit(1)

if __name__ == "__main__":
    compile_resources()
