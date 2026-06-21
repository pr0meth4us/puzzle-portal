import os
import subprocess
import time
import shutil
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SCRIPT_DIR
ARTIFACTS_DIR = Path("/Users/nicksng/.gemini/antigravity-ide/brain/a9d592a8-47fc-46e0-97b5-25fed2b74e19/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PYTHON_PATH = SCRIPT_DIR.parent / "venv" / "bin" / "python"
PORTAL_PATH = SCRIPT_DIR / "portal.py"
RESULT_IMG_PATH = SCRIPT_DIR / "results" / "circled_result.png"

puzzles = [
    {
        "name": "puzzle_02",
        "inputs": [SCRIPT_DIR / "validation_dataset" / "puzzle_02.jpg"],
        "expected": 10
    },
    {
        "name": "puzzle_03",
        "inputs": [SCRIPT_DIR / "validation_dataset" / "puzzle_03.jpg"],
        "expected": 10
    },
    {
        "name": "puzzle_04",
        "inputs": [SCRIPT_DIR / "validation_dataset" / "puzzle_04.jpg"],
        "expected": 10
    },
    {
        "name": "puzzle_05",
        "inputs": [SCRIPT_DIR / "validation_dataset" / "puzzle_05.jpg"],
        "expected": 8
    },
    {
        "name": "puzzle_06",
        "inputs": [SCRIPT_DIR / "validation_dataset" / "puzzle_06.jpg"],
        "expected": 19
    },
    {
        "name": "puzzle_extra_05_vs_06",
        "inputs": [
            SCRIPT_DIR / "puzzles" / "puzzle_extra_05.jpg",
            SCRIPT_DIR / "puzzles" / "puzzle_extra_06.jpg"
        ],
        "expected": 10
    },
    {
        "name": "puzzle_07",
        "inputs": [SCRIPT_DIR / "puzzles" / "puzzle_07.jpg"],
        "expected": 12
    },
    {
        "name": "puzzle_08",
        "inputs": [SCRIPT_DIR / "puzzles" / "puzzle_08.jpg"],
        "expected": 10
    }
]

results = []

print("Starting benchmark for all puzzles...")

for p in puzzles:
    name = p["name"]
    inputs = p["inputs"]
    expected = p["expected"]
    
    print(f"\nProcessing {name}...")
    
    # Copy inputs to artifacts dir
    copied_inputs = []
    for idx, inp in enumerate(inputs):
        ext = inp.suffix
        suffix = f"_in{idx+1}" if len(inputs) > 1 else "_in"
        dest_filename = f"{name}{suffix}{ext}"
        dest_path = ARTIFACTS_DIR / dest_filename
        shutil.copy(inp, dest_path)
        copied_inputs.append(dest_filename)
    
    # Prepare portal command
    cmd = [str(PYTHON_PATH), str(PORTAL_PATH)] + [str(inp) for inp in inputs]
    
    # Run and benchmark
    start_time = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    stdout = res.stdout
    stderr = res.stderr
    
    # Extract details
    category = "Unknown"
    sub_type = "None"
    diff_found = 0
    
    cat_match = re.search(r"Category:\s*(\S+)", stdout)
    if cat_match:
        category = cat_match.group(1)
        
    sub_match = re.search(r"Sub-type:\s*(\S+)", stdout)
    if sub_match:
        sub_type = sub_match.group(1)
        
    diff_match = re.search(r"Differences found\s*:\s*(\d+)", stdout)
    if diff_match:
        diff_found = int(diff_match.group(1))
    else:
        # Check if number grid visual cell diff output
        grid_match = re.search(r"Visual cell differences found:\s*(\d+)", stdout)
        if grid_match:
            diff_found = int(grid_match.group(1))
            
    # Copy output to artifacts dir
    copied_output = None
    if RESULT_IMG_PATH.exists():
        dest_out_path = ARTIFACTS_DIR / f"{name}_out.png"
        shutil.copy(RESULT_IMG_PATH, dest_out_path)
        copied_output = f"{name}_out.png"
        # Remove from local workspace to ensure no crossover
        os.remove(RESULT_IMG_PATH)
        
    results.append({
        "name": name,
        "duration": duration,
        "category": category,
        "sub_type": sub_type,
        "expected": expected,
        "found": diff_found,
        "inputs": copied_inputs,
        "output": copied_output,
        "status": "PASS" if diff_found == expected else "FAIL",
        "stdout_snippet": "\n".join(stdout.split("\n")[-20:])
    })
    
    print(f"  Routed to: Category={category}, Sub-type={sub_type}")
    print(f"  Differences expected: {expected}, Found: {diff_found} ({results[-1]['status']})")
    print(f"  Time taken: {duration:.2f}s")

# Generate Markdown Report
md_content = """# Spot the Difference - Portal Benchmarks and Detections

This document presents the benchmark results and input/output visualizations for all spot-the-difference puzzles routed through the intelligence `portal.py` routing engine.

## Summary Table

| Puzzle Name | Category | Sub-type | Expected Diffs | Found Diffs | Status | Time (s) |
|---|---|---|:---:|:---:|:---:|:---:|
"""

for r in results:
    status_emoji = "✅" if r["status"] == "PASS" else "❌"
    md_content += f"| **{r['name']}** | {r['category']} | {r['sub_type']} | {r['expected']} | {r['found']} | {status_emoji} {r['status']} | {r['duration']:.2f}s |\n"

md_content += "\n---\n\n## Individual Puzzle Details & Visualizations\n\n"

for r in results:
    md_content += f"### {r['name'].replace('_', ' ').title()}\n\n"
    md_content += f"- **Classification**: Category: `{r['category']}`, Sub-type: `{r['sub_type']}`\n"
    md_content += f"- **Benchmark Time**: `{r['duration']:.2f} seconds`\n"
    md_content += f"- **Detections**: Expected `{r['expected']}`, Found `{r['found']}` ({'✅ Matching' if r['status'] == 'PASS' else '❌ Mismatch'})\n\n"
    
    # Input Images
    md_content += "#### Inputs\n"
    if len(r["inputs"]) == 1:
        md_content += f"![{r['name']} Input]({ARTIFACTS_DIR}/{r['inputs'][0]})\n\n"
    else:
        # Carousel for multiple inputs
        md_content += "````carousel\n"
        for idx, inp in enumerate(r["inputs"]):
            md_content += f"![Input {idx+1}]({ARTIFACTS_DIR}/{inp})\n"
            if idx < len(r["inputs"]) - 1:
                md_content += "<!-- slide -->\n"
        md_content += "````\n\n"
        
    # Output Image
    md_content += "#### Detected Differences Output\n"
    if r["output"]:
        md_content += f"![{r['name']} Output]({ARTIFACTS_DIR}/{r['output']})\n\n"
    else:
        md_content += "*No output image generated.*\n\n"
        
    md_content += "---\n\n"

# Write markdown report
with open(ARTIFACTS_DIR / "benchmark_report.md", "w") as f:
    f.write(md_content)

print("\nBenchmark completed. Report written to artifacts/benchmark_report.md")
