import os
import requests
from serpapi import GoogleSearch
from utils.bifrost_config import get_config

def upload_to_temp_host(image_bytes: bytes) -> str:
    """Uploads image bytes to tmpfiles.org and returns the raw image URL."""
    try:
        response = requests.post(
            'https://tmpfiles.org/api/v1/upload',
            files={'file': ('image.jpg', image_bytes, 'image/jpeg')}
        )
        response.raise_for_status()
        data = response.json()
        url = data['data']['url']
        # Convert https://tmpfiles.org/12345/image.jpg to https://tmpfiles.org/dl/12345/image.jpg for raw file
        raw_url = url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
        return raw_url
    except Exception as e:
        print(f"Error uploading image to tmpfiles.org: {e}")
        return ""

def get_google_lens_context(image_bytes: bytes) -> tuple[str, list[bytes]]:
    """Uploads image, queries SerpApi Google Lens, and returns (formatted context string, list of thumbnail image bytes)."""
    api_key = get_config("SERPAPI_KEY")
    if not api_key:
        print("SERPAPI_KEY not found in environment. Please add it to .env!")
        return "", []
        
    image_url = upload_to_temp_host(image_bytes)
    if not image_url:
        return "", []
        
    print(f"🌍 [LENS UPLOAD]: Temporary image URL: {image_url}")
    
    params = {
      "engine": "google_lens",
      "url": image_url,
      "api_key": api_key
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        context_lines = []
        thumbnail_bytes_list = []
        
        if "error" in results:
            print(f"SerpApi Error: {results['error']}")
            return "", []
        
        # Extract Knowledge Graph / Exact match if available
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            kg_title = kg.get("title", "")
            if kg_title:
                context_lines.append(f"Google Lens Knowledge Graph Entity: {kg_title}")
                
        # Extract visual matches and download thumbnails
        if "visual_matches" in results:
            context_lines.append("Google Lens Visual Matches:")
            for match in results["visual_matches"][:4]:  # Top 4 matches
                title = match.get("title", "")
                thumbnail_url = match.get("thumbnail", "")
                
                if title:
                    context_lines.append(f"- {title}")
                
                if thumbnail_url:
                    try:
                        resp = requests.get(thumbnail_url, timeout=5)
                        if resp.status_code == 200:
                            thumbnail_bytes_list.append(resp.content)
                    except Exception as e:
                        print(f"Failed to download thumbnail: {e}")
                        
        return "\n".join(context_lines), thumbnail_bytes_list
        
    except Exception as e:
        print(f"Error querying SerpApi Google Lens: {e}")
        return "", []
