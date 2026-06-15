import os
import requests
from dotenv import load_dotenv

# Load local .env first to get the BIFROST_ credentials
load_dotenv()

# Synchronous Pull on Boot
bifrost_url = os.getenv("BIFROST_URL")
client_id = os.getenv("BIFROST_CLIENT_ID")
webhook_secret = os.getenv("BIFROST_WEBHOOK_SECRET")

if bifrost_url and client_id and webhook_secret:
    endpoint = f"{bifrost_url.rstrip('/')}/api/v1/config"
    headers = {
        "X-Client-ID": client_id,
        "X-Webhook-Secret": webhook_secret
    }
    try:
        response = requests.get(endpoint, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Dump decrypted keys straight into local memory
        keys_loaded = 0
        for key, value in data.get("data", {}).get("api_keys", {}).items():
            if value:
                os.environ[key] = str(value)
                keys_loaded += 1
                
        print(f"✅ Bifrost: Synchronous Pull successful. Loaded {keys_loaded} keys into memory.")
    except Exception as e:
        print(f"❌ Bifrost: Failed to fetch secure config - {e}")
else:
    print("⚠️ Warning: Missing Bifrost credentials. Running with standard local env.")

def get_config(key_name: str, default: str = "") -> str:
    """
    Fetches a config value from the environment.
    (Keys were synchronously injected into os.environ at boot via Bifrost)
    """
    safe_key_name = key_name.strip()
    fallback = os.getenv(safe_key_name)
    if fallback is not None:
        return fallback

    return default
