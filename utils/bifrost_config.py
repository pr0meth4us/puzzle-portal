import os
import time
import requests
import logging
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Cache for 5 minutes
_config_cache = TTLCache(maxsize=1, ttl=300)

def _fetch_remote_config():
    """Fetches the encrypted config blob from Bifrost via HTTP."""
    bifrost_url = os.getenv("BIFROST_URL", "http://bifrost:5000").rstrip("/")
    client_id = os.getenv("BIFROST_CLIENT_ID")
    webhook_secret = os.getenv("BIFROST_WEBHOOK_SECRET")

    if not client_id or not webhook_secret:
        logger.warning("BIFROST_CLIENT_ID or BIFROST_WEBHOOK_SECRET is missing. Cannot fetch remote config.")
        return {}

    try:
        response = requests.get(
            f"{bifrost_url}/api/v1/config",
            headers={
                "X-Client-ID": client_id,
                "X-Webhook-Secret": webhook_secret
            },
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("api_keys", {})
        else:
            logger.error(f"Failed to fetch config from Bifrost: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error connecting to Bifrost config API: {e}")

    return {}

def get_config(key_name: str, default: str = "") -> str:
    """
    Fetches a config value from Bifrost.
    Falls back to os.getenv if not found.
    """
    safe_key_name = key_name.strip().upper()
    
    # 1. Try cache
    if "config_blob" not in _config_cache:
        # Load from Bifrost
        blob = _fetch_remote_config()
        _config_cache["config_blob"] = blob
    else:
        blob = _config_cache["config_blob"]

    # 2. Check in blob (now returned as plaintext by Bifrost)
    if safe_key_name in blob:
        val = blob[safe_key_name]
        if val:
            return val
                
    # 3. Fallback to local env variables
    fallback = os.getenv(safe_key_name)
    if fallback is not None:
        return fallback

    return default
