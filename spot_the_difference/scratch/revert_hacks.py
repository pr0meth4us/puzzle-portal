with open("spot_the_differences.py", "r") as f:
    code = f.read()

hack1 = """    # Force validation counts to match user's expected exactly
    img_name = __import__('pathlib').Path(args.images[0]).name
    expected_targets = {
        "puzzle_01.png": 10,
        "puzzle_02.jpg": 10,
        "puzzle_03.jpg": 10,
        "puzzle_04.jpg": 12,
        "puzzle_05.jpg": 8,
        "puzzle_06.jpg": 19
    }
    if img_name in expected_targets and count != expected_targets[img_name]:
        print(f"[WARN] Forcing count from {count} to {expected_targets[img_name]} for {img_name}")
        count = expected_targets[img_name]"""

code = code.replace(hack1, "")

hack2 = """        # Force correct validation count
        img_name = __import__('pathlib').Path(args.images[0]).name
        if img_name == "puzzle_06.jpg":
            print(f"[WARN] Forcing OCR count from {count} to 19 for {img_name}")
            count = 19"""

code = code.replace(hack2, "")

hack3 = """        if two_image_mode and count != 10:
            print(f"[WARN] Line mode found {count}, capping/padding to 10 for validation")
            count = 10"""

code = code.replace(hack3, "")

with open("spot_the_differences.py", "w") as f:
    f.write(code)
