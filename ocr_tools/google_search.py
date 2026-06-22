#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

# Load secure keys via Bifrost config
from utils.bifrost_config import get_config
from serpapi import GoogleSearch

def copy_to_clipboard(text):
    current_os = platform.system()
    try:
        if current_os == 'Windows':
            subprocess.run(['clip'], input=text.encode('utf-16'), check=True)
        elif current_os == 'Darwin':
            subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)
        elif current_os == 'Linux':
            try:
                subprocess.run(['xclip', '-selection', 'clipboard'], input=text.encode('utf-8'), check=True)
            except FileNotFoundError:
                subprocess.run(['xsel', '--clipboard', '--input'], input=text.encode('utf-8'), check=True)
        else:
            print(f"  [clipboard not supported on OS: {current_os}]")
    except Exception as e:
        print(f"  [clipboard failed: {e}]")

def sync_to_icloud(text):
    paths = [
        Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/clip.txt",
        Path.home() / "Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/clip.txt"
    ]
    for p in paths:
        try:
            if p.parent.exists():
                p.write_text(text, encoding="utf-8")
                print(f"Synced to iCloud: {p.name} ({p.parent.name})")
        except Exception as e:
            print(f"  [iCloud sync failed for {p.parent.name}: {e}]")

def sync_to_local_server(text):
    try:
        local_file = Path("/tmp/local_clip.txt")
        local_file.write_text(text, encoding="utf-8")
        
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.2)
        result = s.connect_ex(('127.0.0.1', 8089))
        s.close()
        
        if result != 0:
            server_script = SCRIPT_DIR / "clip_server.py"
            subprocess.Popen([sys.executable, str(server_script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Local clip server started on port 8089.")
        else:
            print("Local clip server updated.")
    except Exception as e:
        print(f"  [Local server update failed: {e}]")

def main():
    start_time = time.time()
    if len(sys.argv) < 2:
        print("Usage: ./google_search.py <query>")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    
    api_key = get_config("SERPAPI_KEY")
    if not api_key:
        print("ERROR: SERPAPI_KEY not found in environment or Bifrost config!")
        sys.exit(1)
        
    print(f"🔍 Searching Google for: '{query}'...")
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
    except Exception as e:
        print(f"ERROR: SerpApi request failed: {e}")
        sys.exit(1)
        
    if "error" in results:
        print(f"SerpApi Error: {results['error']}")
        sys.exit(1)
        
    answer = None
    source_type = None
    
    # 1. Check AI Overview (Google's generative overview)
    if "ai_overview" in results:
        ai_ov = results["ai_overview"]
        # If it's a stub containing serpapi_link, fetch the full details
        if "serpapi_link" in ai_ov and "text_blocks" not in ai_ov:
            try:
                link = ai_ov["serpapi_link"] + f"&api_key={api_key}"
                response = requests.get(link, timeout=10)
                response.raise_for_status()
                ai_data = response.json()
                if "ai_overview" in ai_data:
                    ai_ov = ai_data["ai_overview"]
            except Exception as e:
                print(f"Warning: Failed to fetch full AI Overview from SerpApi link: {e}")

        if "text_blocks" in ai_ov:
            blocks = []
            for b in ai_ov["text_blocks"]:
                if b.get("type") == "list":
                    for item in b.get("list", []):
                        if item.get("snippet"):
                            blocks.append(f"- {item.get('snippet')}")
                else:
                    if b.get("snippet"):
                        blocks.append(b.get("snippet"))
            if blocks:
                answer = "\n\n".join(blocks)
                source_type = "AI Overview"

    # 2. Check Answer Box (Direct Answer)
    if not answer and "answer_box" in results:
        ab = results["answer_box"]
        answer = ab.get("answer") or ab.get("snippet") or ab.get("result")
        if answer:
            source_type = "Answer Box"
            
    # 3. Check Knowledge Graph
    if not answer and "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        answer = kg.get("description") or kg.get("snippet")
        if answer:
            source_type = "Knowledge Graph"
            
    # 4. Fallback to Organic Results snippet
    if not answer and "organic_results" in results and len(results["organic_results"]) > 0:
        top_result = results["organic_results"][0]
        answer = top_result.get("snippet")
        if answer:
            source_type = f"Organic Result #1 ({top_result.get('title', 'Unknown Source')})"
            
    duration = time.time() - start_time
    if answer:
        print(f"\n✨ Found answer via [{source_type}]:")
        print(answer)
        
        first_line = answer.splitlines()[0].strip() if answer.strip() else ""
        copy_to_clipboard(first_line)
        
        # Audible alert
        if platform.system() == 'Darwin':
            try:
                subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'])
            except Exception:
                pass
                
        # Big visual alert
        print("\n\033[42;97;1m ✨✨ SUCCESS: COPIED & SYNCED! ✨✨ \033[0m")
        print(f"Copied first line: '{first_line}'\n")

        sync_to_icloud(first_line)
        sync_to_local_server(first_line)
    else:
        print("\n❌ No clear answer or snippets found in results.")
        if "organic_results" in results:
            print("Top Organic Results:")
            for i, res in enumerate(results["organic_results"][:3]):
                print(f"{i+1}. {res.get('title')} ({res.get('link')})")
        else:
            print("Results keys:", list(results.keys()))
            
    print(f"\nTime: {duration:.3f}s")

if __name__ == '__main__':
    main()
