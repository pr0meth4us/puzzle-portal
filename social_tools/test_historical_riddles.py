#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import asyncio
from pathlib import Path
from bs4 import BeautifulSoup
from google import genai
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Stealth plugin to patch browser fingerprint leaks
try:
    from playwright_stealth import stealth_async
except ImportError:
    stealth_async = None

# Load environment variables
load_dotenv_script = """
import os
from pathlib import Path
from dotenv import load_dotenv
parent_dir = Path(__file__).resolve().parent.parent
dotenv_path = parent_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)
"""
exec(load_dotenv_script)

HTML_FILE_PATH = "/Users/nicksng/Desktop/facebook-spnn17-06_06_2026-VSBwAAMW/your_facebook_activity/messages/inbox/1276957488995520/message_1.html"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').replace('\ufeff', '').strip()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "expTracker")

SCRIPT_DIR = Path(__file__).resolve().parent
SESSION_DIR = SCRIPT_DIR / "fb_session"
STATE_FILE = SCRIPT_DIR / "fb_state.json"

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY missing from .env")
    sys.exit(1)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

def _human_delay(min_ms: int = 800, max_ms: int = 2500):
    return random.randint(min_ms, max_ms)

def load_state_from_db() -> str | None:
    if not MONGODB_URI: return None
    try:
        from pymongo import MongoClient
        print("Connecting to MongoDB to fetch Facebook state...")
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        doc = db["fb_settings"].find_one({"key": "state_json"})
        if doc and "value" in doc:
            print("Successfully retrieved Facebook state from MongoDB.")
            return doc["value"]
    except Exception as e:
        print(f"Warning: Failed to fetch Facebook state from MongoDB: {e}")
    return None

def save_state_to_db(state_json: str) -> None:
    if not MONGODB_URI: return
    try:
        from pymongo import MongoClient
        print("Connecting to MongoDB to save Facebook state...")
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        db["fb_settings"].update_one(
            {"key": "state_json"},
            {"$set": {"value": state_json, "updated_at": time.time()}},
            upsert=True
        )
        print("Successfully saved Facebook state to MongoDB.")
    except Exception as e:
        print(f"Warning: Failed to save Facebook state to MongoDB: {e}")

def extract_riddles_from_html():
    print(f"Reading {HTML_FILE_PATH}...")
    try:
        with open(HTML_FILE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading HTML: {e}")
        return []

    soup = BeautifulSoup(content, 'html.parser')
    sections = soup.find_all('section', class_='_a6-g')
    riddles = []
    
    for section in sections:
        sender_tag = section.find('h2')
        if not sender_tag: continue
        sender = sender_tag.get_text(strip=True)
        if sender != "Ron Oum": continue
            
        content_div = section.find('div', class_='_a6-p')
        if not content_div: continue
            
        for br in content_div.find_all("br"):
            br.replace_with("\n")
            
        text = content_div.get_text(separator=' ', strip=True)
        if "សំណួរ" in text or "ឆ្លើយត្រូវមាន" in text:
            riddles.append(text)
            
    print(f"Found {len(riddles)} riddles from Ron Oum.")
    return riddles

async def get_gemini_answer(riddle_text):
    # Load family context
    try:
        with open(SCRIPT_DIR / "family_facts.csv", "r", encoding="utf-8") as f:
            family_csv = f.read()
        with open(SCRIPT_DIR / "family_relations.tsv", "r", encoding="utf-8") as f:
            family_tsv = f.read()
        family_context = f"Family Prizes:\n{family_csv}\n\nFamily Relationships:\n{family_tsv}"
    except Exception:
        family_context = ""

    prompt = (
        "You are an intelligent assistant playing a fast-paced riddle game in a Khmer group chat. Here is a riddle or trivia question:\n\n"
        f"\"{riddle_text}\"\n\n"
        "Provide ONLY the correct, direct answer in Khmer. DO NOT be conversational. DO NOT explain your reasoning. JUST answer the riddle. "
        "Keep the answer as short as possible.\n\n"
        f"Family Context:\n{family_context}"
    )
    try:
        response = await genai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error getting answer: {e}"

async def take_screenshot(page, step_name: str):
    ss_path = os.path.abspath(f"fb_debug_{step_name}.png")
    try:
        await page.screenshot(path=ss_path)
        print(f"SCREENSHOT_SAVED:{ss_path}")
    except Exception as e:
        print(f"Warning: Failed to capture screenshot {step_name}: {e}")

async def send_to_self(answers_dict, headed=True):
    print("\n=== Playwright: Sending Answers to Self ===")
    
    # Restore state from DB if available
    state_json_content = load_state_from_db()
    if state_json_content:
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                f.write(state_json_content.strip())
            print("Successfully restored session context locally from MongoDB.")
        except Exception:
            pass

    async with async_playwright() as p:
        launch_args = ["--disable-blink-features=AutomationControlled"]
        if not headed:
            launch_args.extend(["--no-sandbox", "--disable-dev-shm-usage", "--disable-setuid-sandbox"])
            
        browser = await p.chromium.launch(
            headless=not headed,
            args=launch_args,
            ignore_default_args=["--enable-automation"]
        )
        
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        
        # Load state if exists
        context_options = {
            "viewport": {"width": 1280, "height": 800},
            "user_agent": user_agent,
            "locale": "en-US"
        }
        if STATE_FILE.exists():
            context_options["storage_state"] = STATE_FILE

        context = await browser.new_context(**context_options)
        page = await context.new_page()

        if stealth_async and not headed:
            await stealth_async(page)

        try:
            print("Navigating to Messenger...")
            await page.goto("https://www.messenger.com/", wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(_human_delay(3000, 5000))
            
            # Check login
            if "login" in page.url or await page.locator("input[name='email']").is_visible():
                if not headed:
                    print("ERROR: Login required. Please run headed mode to login.")
                    await take_screenshot(page, "login_required")
                    return
                print("Please log into Messenger in the browser window.")
                print("Waiting up to 60 seconds for login to complete...")
                try:
                    await page.wait_for_selector('a[aria-label="New message"], div[role="navigation"]', timeout=60000)
                    print("Login detected.")
                    
                    # Save state back to local and DB
                    await context.storage_state(path=STATE_FILE)
                    with open(STATE_FILE, "r", encoding="utf-8") as f:
                        save_state_to_db(f.read())
                except PlaywrightTimeoutError:
                    print("Timeout waiting for login. Proceeding anyway, might fail.")
            
            # Verification of session
            is_session_valid = False
            try:
                new_msg_btn = page.locator('a[aria-label="New message"]').first
                await new_msg_btn.wait_for(state="visible", timeout=15000)
                is_session_valid = True
                await new_msg_btn.click()
                await page.wait_for_timeout(_human_delay(1500, 3000))
                
                search_input = page.locator('input[aria-label="Search for people and groups"]').first
                await search_input.fill("Nick Sng")
                await page.wait_for_timeout(_human_delay(2000, 4000))
                
                # Use fallback selectors
                first_result = page.locator('ul[role="listbox"] li, [role="option"]').first
                if await first_result.is_visible():
                    await first_result.click()
                    await page.wait_for_timeout(_human_delay(1500, 3000))
                else:
                    print("Could not find yourself in the search.")
                    await take_screenshot(page, "search_failed")
                    return
                    
            except Exception as e:
                print(f"Error navigating to self-chat: {e}")
                await take_screenshot(page, "nav_failed")
                
            if is_session_valid:
                # Type answers
                text_input = page.locator('[role="textbox"][contenteditable="true"]').first
                if await text_input.is_visible():
                    for riddle, answer in answers_dict.items():
                        message_text = f"RIDDLE:\n{riddle}\n\nANSWER:\n{answer}"
                        print(f"Sending:\n{message_text}\n---")
                        
                        await text_input.click()
                        await page.wait_for_timeout(_human_delay(500, 1000))
                        
                        # Simulate human typing
                        await text_input.press_sequentially(message_text, delay=random.randint(20, 60))
                        await page.wait_for_timeout(_human_delay(800, 1500))
                        await text_input.press("Enter")
                        await page.wait_for_timeout(_human_delay(2000, 4000))
                        
                    print("Successfully sent answers to yourself!")
                else:
                    print("ERROR: Chat input box not found.")
                    await take_screenshot(page, "input_not_found")
            else:
                print("⚠️ Session invalid (logged out or flagged). Skipping state save.")

        except PlaywrightTimeoutError as err:
            print(f"❌ Automation timed out: {err}")
            await take_screenshot(page, "timeout_error")
        except Exception as err:
            print(f"❌ An error occurred: {err}")
            await take_screenshot(page, "general_error")
        finally:
            print("\nClosing browser...")
            await browser.close()

async def main():
    riddles = extract_riddles_from_html()
    if not riddles:
        print("No riddles found.")
        return
        
    riddles = riddles[:10]
    answers_dict = {}
    print(f"\n--- Getting Answers from Gemini for {len(riddles)} riddles ---")
    for r in riddles:
        print(f"Riddle: {r[:100]}...")
        ans = await get_gemini_answer(r)
        print(f"Answer: {ans}")
        answers_dict[r] = ans
        
    await send_to_self(answers_dict, headed=True)

if __name__ == "__main__":
    asyncio.run(main())
