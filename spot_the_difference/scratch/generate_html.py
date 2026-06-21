import base64
from pathlib import Path

html = "<html><body><h1>Puzzles and Answers</h1>"
val_dir = Path("validation_dataset")

for i in range(1, 7):
    p_name = f"puzzle_0{i}.jpg" if i != 1 else "puzzle_01.png"
    a_name = f"answer_0{i}.jpg"
    
    html += f"<h2>Puzzle {i}</h2>"
    
    p_path = val_dir / p_name
    if p_path.exists():
        with open(p_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
            ext = p_name.split('.')[-1]
            html += f"<img src='data:image/{ext};base64,{b64}' width='400'>"
            
    a_path = val_dir / a_name
    if a_path.exists():
        with open(a_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
            html += f"<img src='data:image/jpeg;base64,{b64}' width='400'>"
            
html += "</body></html>"

with open("../artifacts/puzzles.html", "w") as f:
    f.write(html)
print("Generated puzzles.html in artifacts")
