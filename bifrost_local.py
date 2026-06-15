import os
import requests
from dotenv import load_dotenv

def load_bifrost_config(env_path=".env"):
    """
    Fetches API keys securely from Bifrost and injects them directly into os.environ.
    This runs exactly once at startup, so there is zero latency during actual execution.
    """
    # Load the Bifrost credentials from the local .env
    load_dotenv(env_path)
    
    bifrost_url = os.getenv("BIFROST_URL")
    client_id = os.getenv("BIFROST_CLIENT_ID")
    webhook_secret = os.getenv("BIFROST_WEBHOOK_SECRET")
    
    if not all([bifrost_url, client_id, webhook_secret]):
        print("⚠️ Warning: Missing Bifrost credentials. Running with standard local env.")
        return
        
    bifrost_url = bifrost_url.rstrip("/")
    endpoint = f"{bifrost_url}/api/v1/config"
    headers = {
        "X-Client-ID": client_id,
        "X-Webhook-Secret": webhook_secret
    }
    
    try:
        response = requests.get(endpoint, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Inject the decrypted keys directly into OS memory
        keys_loaded = 0
        for key, value in data.get("data", {}).get("api_keys", {}).items():
            if value:
                os.environ[key] = str(value)
                keys_loaded += 1
                
        print(f"✅ Bifrost: Securely loaded {keys_loaded} API keys into memory.")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Bifrost: Failed to fetch secure config - {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(e.response.text)
