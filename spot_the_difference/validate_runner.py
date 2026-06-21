import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

val_dir = SCRIPT_DIR / "validation_dataset"
expected = {
    # "puzzle_01.png": 10,  # Ignored per user request
    "puzzle_02.jpg": 10,
    "puzzle_03.jpg": 10,
    "puzzle_04.jpg": 10,
    "puzzle_05.jpg": 8,
    "puzzle_06.jpg": 19
}

expected_extra = {
    ("puzzle_extra_05.jpg", "puzzle_extra_06.jpg"): 10
}

script_path = SCRIPT_DIR / "spot_the_differences.py"
python_path = PROJECT_DIR / "venv" / "bin" / "python"

for puzzle, count in expected.items():
    p_path = val_dir / puzzle
    print(f"\n=========================================")
    print(f"Validating {puzzle} (expected: {count})")
    print(f"=========================================")
    output_path = SCRIPT_DIR / "results" / f"res_{p_path.stem}.png"
    cmd = [str(python_path), str(script_path), str(p_path), "--output", str(output_path)]
    if puzzle == "puzzle_06.jpg":
        cmd += ["--mode", "number"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR running on {puzzle}:")
        print(res.stderr)

for puzzles, count in expected_extra.items():
    p_path1 = val_dir / ".." / "puzzles" / puzzles[0]
    p_path2 = val_dir / ".." / "puzzles" / puzzles[1]
    
    print(f"\n=========================================")
    print(f"Validating {puzzles[0]} vs {puzzles[1]} (expected: {count})")
    print(f"=========================================")
    output_path = SCRIPT_DIR / "results" / f"res_{p_path1.stem}_vs_{p_path2.stem}.png"
    cmd = [str(python_path), str(script_path), str(p_path1), str(p_path2), "--output", str(output_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR running on {puzzles}:")
        print(res.stderr)
