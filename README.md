# puzzle-portal

AI-powered puzzle portal and social worker bot. Classified, routes, and solves visual puzzles, number grids, and spot-the-difference images, then posts/interacts via social channels.

---

- **puzzle_portal** — AI router to classify puzzles and dispatch to spot_the_difference/ocr_tools workers
- **spot_the_difference** — opencv thing that finds differences between two images
- **ocr_tools** — reads khmer text from images, asks gemini what it means
- **social_tools** — tiktok bot and facebook messenger bots that text people so i don't have to
- **object_detection** — counts and labels rabbits and birds in optical illusions using YOLO/OwlViT/Gemini

---

## setup

```bash
pip install -r requirements.txt
playwright install chromium  # only if you're using the tiktok thing
```

copy `.env` and fill in whatever keys the thing you're running needs (`GEMINI_API_KEY`, `BIFROST_URL`, `BIFROST_CLIENT_ID`, `BIFROST_WEBHOOK_SECRET`, etc.)

