import json
import requests

def fetch_json(url, params=None, headers=None, timeout=30):
    """Fetch JSON from an HTTP endpoint and return (data, response).

    Raises a requests exception if the request fails.
    """
    default_headers = {
        "User-Agent": "NOVA-IMS-Data-Stream-Mining-Practical-Assignment-1"
    }

    if headers:
        default_headers.update(headers)

    response = requests.get(
        url,
        params=params,
        headers=default_headers,
        timeout=timeout
    )
    response.raise_for_status()

    return response.json(), response

def inspect_json(obj, name="root", max_items=5):
    """Small helper for inspecting an unfamiliar JSON object."""
    print(f"{name}: {type(obj).__name__}")

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("Keys:", keys[:max_items])
        if len(keys) > max_items:
            print(f"... {len(keys) - max_items} more keys")

    elif isinstance(obj, list):
        print("Length:", len(obj))
        if obj:
            print("First item type:", type(obj[0]).__name__)
            print("First item:", obj[0])