import os
import urllib.request
import json
from pathlib import Path

def main():
    env_file = Path('d:/Magang/rag/backend/.env')
    api_key = None
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('GOOGLE_API_KEY='):
                    api_key = line.strip().split('=', 1)[1].strip('"\'')
                    break
    
    if not api_key:
        print("API Key not found")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    try:
        data = json.dumps({'model': 'models/gemini-embedding-001', 'content': {'parts': [{'text': 'hello world'}]}}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"Dimension: {len(result['embedding']['values'])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
