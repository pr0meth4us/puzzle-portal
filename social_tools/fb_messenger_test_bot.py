#!/usr/bin/env python3
import os
import sys
import time
import random
import asyncio
import certifi
from pathlib import Path
from google import genai
from google.genai import types
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from google.cloud import vision
from utils.bifrost_config import get_config

# Stealth plugin
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

GEMINI_API_KEY = get_config('GEMINI_API_KEY', '').replace('\ufeff', '').strip()
MONGODB_URI = get_config("MONGODB_URI")
DB_NAME = get_config("DB_NAME", "bifrost")

SCRIPT_DIR = Path(__file__).resolve().parent
SESSION_DIR = SCRIPT_DIR / "fb_session_test_bot"
STATE_FILE = SCRIPT_DIR / "fb_state_test_bot.json"

TARGET_URL = "https://www.messenger.com/e2ee/t/8591487297628701/"

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY missing from .env")
    sys.exit(1)

# Initialize Clients
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Set up Google Cloud Vision
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'False'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = get_config(
    'GOOGLE_APPLICATION_CREDENTIALS',
    str(SCRIPT_DIR.parent / 'credentials.json')
)
vision_client = vision.ImageAnnotatorClient()

def _human_delay(min_ms: int = 800, max_ms: int = 2500):
    return random.randint(min_ms, max_ms)

def load_state_from_db() -> str | None:
    if not MONGODB_URI: return None
    try:
        from pymongo import MongoClient
        print("Connecting to MongoDB to fetch Facebook state...")
        client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        doc = db["fb_settings"].find_one({"key": "state_json_bot"})
        if doc and "value" in doc:
            return doc["value"]
    except Exception as e:
        print(f"Warning: Failed to fetch Facebook state from MongoDB: {e}")
    return None

def save_state_to_db(state_json: str) -> None:
    if not MONGODB_URI: return
    try:
        from pymongo import MongoClient
        print("Connecting to MongoDB to save Facebook state...")
        client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = client[DB_NAME]
        db["fb_settings"].update_one(
            {"key": "state_json_bot"},
            {"$set": {"value": state_json, "updated_at": time.time()}},
            upsert=True
        )
        print("Successfully saved Facebook state to MongoDB.")
    except Exception as e:
        print(f"Warning: Failed to save Facebook state to MongoDB: {e}")

# Context string for family-specific riddles
try:
    with open(SCRIPT_DIR / "family_facts.csv", "r", encoding="utf-8") as f:
        FAMILY_CONTEXT_CSV = f.read()
    with open(SCRIPT_DIR / "family_relations.tsv", "r", encoding="utf-8") as f:
        FAMILY_RELATIONS_TSV = f.read()
        
    FAMILY_CONTEXT = (
        f"The family members and their riddle prize winnings are detailed in this CSV data:\n{FAMILY_CONTEXT_CSV}\n\n"
        f"The relationships of the family members to 'Oum Soeung' and 'Mom Tangyue' are detailed in this TSV data:\n{FAMILY_RELATIONS_TSV}"
    )
except Exception as e:
    FAMILY_CONTEXT = "The family members include: Ron Oum, Neternal Soeung, Sopheara Oum, Loeum Khay, etc."

MY_NAME = "Nick" # Update this to your exact Facebook name used in the chat

chat_history = []

async def get_gemini_answer_from_screenshot(image_bytes):
    global chat_history
    
    system_instruction = (
        "You are an AI playing a fast-paced riddle game in a Khmer group chat. "
        "You will receive screenshots of new messages one by one. "
        "Riddles can be text, 'spot the difference', math, counting animals, or family trivia. "
        "Sometimes a riddle spans multiple messages (e.g., text then an image). Remember previous messages to piece it together. "
        "Rules:\n"
        "1. If the NEWEST message completes the CURRENT riddle or is a continuation, solve it and provide ONLY the short, direct answer in Khmer.\n"
        f"2. If the NEWEST message announces the winners of a riddle and MY NAME ({MY_NAME}) is in the winners list, reply EXACTLY with 'WON'.\n"
        "3. If winners are announced but I am not on the list, or the puzzle is clearly over, reply EXACTLY with 'RESET'.\n"
        "4. If the NEWEST message is a COMPLETELY NEW riddle, solve it and reply EXACTLY with 'NEW_RIDDLE_ANSWER: [your answer]'.\n"
        "5. If the newest message is NOT a riddle (e.g. it is a greeting, normal conversation, questions like 'how are you?'), or if you ALREADY answered the riddle, you MUST reply EXACTLY with 'IGNORE'. DO NOT chat. DO NOT respond to conversation. YOU ONLY ANSWER RIDDLES.\n"
        "6. CRITICAL: Many images in this chat are just generic photos, selfies, memes, or unrelated screenshots. If the image is NOT clearly a puzzle, riddle, or game, you MUST output EXACTLY: IGNORE\n"
        "7. Do NOT hallucinate puzzles. Do NOT route a generic photo of a tree or beach to Spot the Difference.\n"
        "8. **CRITICAL**: We are NO LONGER solving visual puzzles (Spot the Difference, count animals, optical illusions) automatically. If the image is a NEW visual puzzle, you MUST output EXACTLY: 'IGNORE' to hold off and wait.\n"
        "9. IF the NEWEST message is ANOTHER HUMAN USER in the chat answering the visual puzzle (e.g., they say '7 differences', '5', or post a solved image), you MUST steal their answer! Output EXACTLY: 'NEW_RIDDLE_ANSWER: [their text answer]'. If their answer is just a circled image, output EXACTLY: 'FORWARD_IMAGE'.\n"
        "10. If it is a genuine math puzzle, logic puzzle, or text riddle that you can solve right now, output your thought process first, then output the final concise answer. DO NOT be conversational. JUST answer the riddle.\n"
        "11. Do NOT chat, talk, or make conversation with the users in the group chat. If the text does not contain a riddle or an answer to a visual puzzle to copy, you MUST output IGNORE.\n"
        "12. **DEBUG**: Before you output your final answer (like IGNORE or NEW_RIDDLE_ANSWER: xxx), you MUST print a line starting with 'REASONING: ' explaining what you see in the image and why you are choosing that answer.\n"
        f"Family Context: {FAMILY_CONTEXT}"
    )
    
    try:
        # Step 1: Run Cloud Vision OCR
        v_image = vision.Image(content=image_bytes)
        v_request = {
            'image': v_image,
            'features': [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}]
        }
        v_response = vision_client.annotate_image(v_request)
        ocr_text = v_response.full_text_annotation.text if v_response.full_text_annotation else ""
        
        # Step 2: Query Gemini
        fast_parts = []
        if ocr_text:
            fast_parts.append(types.Part.from_text(text=f"OCR Extracted Text:\n{ocr_text}\n\nImage:"))
        else:
            fast_parts.append(types.Part.from_text(text="Image:"))
        fast_parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        
        current_turn = types.Content(role="user", parts=fast_parts)
        temp_history = chat_history + [current_turn]
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1
        )
        
        response = await genai_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=temp_history,
            config=config
        )
        answer = response.text.strip()
        print(f"🧠 [GEMINI RAW]:\n{answer}")
        
        # Extract the actual command from the output
        final_answer = "IGNORE"
        if "NEW_RIDDLE_ANSWER:" in answer:
            final_answer = "NEW_RIDDLE_ANSWER:" + answer.split("NEW_RIDDLE_ANSWER:")[1].split('\n')[0].strip()
        elif "FORWARD_IMAGE" in answer:
            final_answer = "FORWARD_IMAGE"
        else:
            for line in answer.split('\n'):
                if line.startswith("WON") or line.startswith("RESET") or line == "IGNORE":
                    final_answer = line.strip()
                
        answer = final_answer
        print(f"🎯 [PARSED COMMAND]: {answer}")
        
        if answer == "FORWARD_IMAGE":
            import time
            out_img_path = f"/tmp/copied_answer_{int(time.time()*1000)}.jpg"
            with open(out_img_path, "wb") as f:
                f.write(image_bytes)
            answer = f"NEW_RIDDLE_ANSWER: [IMAGE_RESPONSE: {out_img_path}]"
            print(f"🎯 [COPYCAT]: Stealing human's image answer and saving to {out_img_path}")
            
        # History management
        if answer.startswith("NEW_RIDDLE_ANSWER:"):
            actual_answer = answer.replace("NEW_RIDDLE_ANSWER:", "").strip()
            # Clear old history and only keep this new riddle's context
            chat_history.clear()
            chat_history.append(current_turn)
            model_turn = types.Content(role="model", parts=[types.Part.from_text(text=actual_answer)])
            chat_history.append(model_turn)
            return answer
        
        # Append the user turn since it was successfully processed
        chat_history.append(current_turn)
        
        # Save model's reply to history so it remembers what it said
        model_turn = types.Content(role="model", parts=[types.Part.from_text(text=answer)])
        chat_history.append(model_turn)
        
        # Keep history size manageable (last 10 turns = 5 user, 5 model)
        if len(chat_history) > 10:
            chat_history[:] = chat_history[-10:]
        
        return answer
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "IGNORE"

async def take_screenshot(page, step_name: str):
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    ss_path = log_dir / f"fb_bot_debug_{step_name}.png"
    try:
        await page.screenshot(path=ss_path)
        print(f"SCREENSHOT_SAVED:{ss_path}")
    except Exception as e:
        print(f"Warning: Failed to capture screenshot {step_name}: {e}")

async def run_bot(headed=True):
    print("\n=== Playwright: Facebook Messenger TEST Bot ===")
    
    # IGNORING DB STATE: We want to run on Nick's main account!
    # We will just let Playwright use the existing fb_state_bot.json

    async with async_playwright() as p:
        launch_args = ["--disable-blink-features=AutomationControlled"]
        if not headed:
            launch_args.extend(["--no-sandbox", "--disable-dev-shm-usage", "--disable-setuid-sandbox"])
            
        browser = await p.chromium.launch(
            headless=not headed,
            channel="chrome",
            args=launch_args,
            ignore_default_args=["--enable-automation"]
        )
        
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        
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

        seen_messages = set()

        try:
            print(f"Navigating to Messenger...")
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
                    
                    # Save state locally so we don't have to log in again!
                    await context.storage_state(path=STATE_FILE)
                    print("Local session state saved.")
                except PlaywrightTimeoutError:
                    print("Timeout waiting for login. Proceeding anyway, might fail.")
            
            # Global check for PIN modal right after loading the main page (it often blocks the whole screen)
            try:
                pin_text = page.get_by_text("Enter your PIN", exact=False).first
                await pin_text.wait_for(state="visible", timeout=5000)
                print("Found global PIN modal. Entering PIN '433764'...")
                await page.keyboard.insert_text("433764")
                await page.wait_for_timeout(3000)
                
                # Save state again after PIN is entered to remember the device!
                await context.storage_state(path=STATE_FILE)
                print("Local session state updated after PIN.")
            except Exception:
                pass

            is_session_valid = False
            try:
                print(f"Successfully connected to the target chat!")
                
                # Check for the "Continue" button (End-to-End Encryption upgrade prompt)
                continue_btn = page.locator('div[role="button"], button').filter(has_text="Continue").first
                if await continue_btn.is_visible():
                    print("Found 'Continue' button for E2E encryption. Clicking it...")
                    await continue_btn.click()
                    await page.wait_for_timeout(_human_delay(2000, 4000))

                # Check for the "Enter your PIN" modal
                try:
                    pin_text = page.get_by_text("Enter your PIN", exact=False).first
                    # Wait up to 5 seconds for the modal to animate/appear
                    await pin_text.wait_for(state="visible", timeout=5000)
                    print("Found PIN modal. Entering PIN '433764'...")
                    await page.keyboard.insert_text("433764")
                    await page.wait_for_timeout(3000)
                except Exception:
                    print("No PIN modal detected within 5 seconds, continuing...")

                # Check if chat input is visible to confirm we're in the chat
                text_input = page.locator('[role="textbox"][contenteditable="true"]').first
                if await text_input.is_visible():
                    is_session_valid = True
                else:
                    print(f"Could not load chat for '{TARGET_URL}'.")
                    await take_screenshot(page, "chat_load_failed")
                    return
            except Exception as e:
                print(f"Error navigating to target chat: {e}")
                await take_screenshot(page, "nav_failed")
                return
                
            if not is_session_valid:
                print("⚠️ Session invalid (logged out or flagged).")
                return

            print(f"Monitoring chat with '{TARGET_URL}' for riddles. Press Ctrl+C to stop.")
            
            is_startup = False
            seen_msg_ids = set()
            
            text_input = page.locator('[role="textbox"][contenteditable="true"]').first
            if not await text_input.is_visible():
                print("ERROR: Chat input box not found.")
                await take_screenshot(page, "input_not_found")
                return
                
            # DUMP THE DOM FOR INSPECTION
            print("DEBUG: Dumping E2EE chat DOM for inspection...")
            try:
                # Try to get the main chat container
                main_area = page.locator('div[role="main"]')
                dump_path = SCRIPT_DIR.parent / "chat_tools" / "e2ee_chat_dump.html"
                if await main_area.is_visible():
                    html_content = await main_area.inner_html()
                    with open(dump_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    print(f"DEBUG: Successfully dumped DOM to {dump_path}")
                else:
                    # Fallback to entire body
                    html_content = await page.inner_html('body')
                    with open(dump_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    print(f"DEBUG: Dumped entire body DOM to {dump_path}")
            except Exception as e:
                print(f"Error dumping DOM: {e}")

            while True:
                try:
                    # Look at the bottom 5 message containers inside the main chat area
                    all_rows = await page.locator('div[role="main"] div[role="row"], div[role="main"] div[data-scope="messages_table"]').all()
                    if not all_rows:
                        # print("DEBUG: Can't find any messages! (E2EE chats might hide them from the bot).")
                        await page.wait_for_timeout(3000)
                        continue
                        
                    bottom_rows = all_rows[-5:]
                    
                    for row in bottom_rows:
                        # Extract a stable unique ID for the message (handles React DOM rebuilds perfectly)
                        msg_id = await row.evaluate('''(node) => {
                            let el = node.querySelector('[id^="mid."]');
                            if (el) return el.id;
                            return node.innerText.substring(0, 100); // Fallback
                        }''')
                        
                        if msg_id not in seen_msg_ids:
                            seen_msg_ids.add(msg_id)
                            
                            if not is_startup:
                                print(f"\n[NEW MESSAGE DETECTED] ID: {msg_id}")
                                
                                try:
                                    # Ensure the row is in view before screenshotting
                                    await row.scroll_into_view_if_needed()
                                    screenshot_bytes = await row.screenshot()
                                    
                                    print("Analyzing message bubble with Gemini...")
                                    answer = await get_gemini_answer_from_screenshot(screenshot_bytes)
                                    
                                    if answer == "WON":
                                        print("[GEMINI]: We won! Saying thanks and clearing history...")
                                        global chat_history
                                        chat_history.clear()
                                        await text_input.click()
                                        await page.wait_for_timeout(500)
                                        await page.keyboard.insert_text("អរគុណច្រើនបង! ជយោ!")
                                        await page.wait_for_timeout(500)
                                        await text_input.press("Enter")
                                        await page.wait_for_timeout(2000)
                                    elif answer == "RESET":
                                        print("[GEMINI]: Puzzle ended or we lost. Clearing history...")
                                        chat_history.clear()
                                    elif answer.startswith("NEW_RIDDLE_ANSWER:"):
                                        answer = answer.replace("NEW_RIDDLE_ANSWER:", "").strip()
                                        print(f"[GEMINI]: Detected a completely new riddle context!")
                                        print(f"[GEMINI ANSWER]: {answer}")
                                        print("Typing and sending...")
                                        await text_input.click()
                                        await page.wait_for_timeout(500)
                                        await page.keyboard.insert_text(answer)
                                        await page.wait_for_timeout(500)
                                        await text_input.press("Enter")
                                        await page.wait_for_timeout(2000)
                                    elif answer != "IGNORE" and answer:
                                        print(f"[GEMINI ANSWER]: {answer}")
                                        print("Typing and sending...")
                                        await text_input.click()
                                        await page.wait_for_timeout(500)
                                        await page.keyboard.insert_text(answer)
                                        await page.wait_for_timeout(500)
                                        await text_input.press("Enter")
                                        await page.wait_for_timeout(2000)
                                    else:
                                        print("[GEMINI]: Ignored (Not a riddle).")
                                except Exception as e:
                                    print(f"Error processing specific message bubble: {e}")
                    
                    if is_startup:
                        is_startup = False
                        
                except Exception as e:
                    print(f"Error checking messages: {e}")
                
                # Poll every 3 seconds
                await page.wait_for_timeout(3000)

        except PlaywrightTimeoutError as err:
            print(f"❌ Automation timed out: {err}")
            await take_screenshot(page, "timeout_error")
        except Exception as err:
            print(f"❌ An error occurred: {err}")
            await take_screenshot(page, "general_error")
        except KeyboardInterrupt:
            print("\nStopping bot...")
        finally:
            print("\nClosing browser...")
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run_bot(headed=True))
