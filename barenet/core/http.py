# Auto-generated on 2026-01-07T17:18:17.769972Z

import requests
from urllib.parse import urlparse

class PassiveHTTPClient:
    def fetch(self, url):
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            raise ValueError('Invalid scheme')
        r = requests.get(url, timeout=5)
        return type('Resp', (), {
            'url': r.url,
            'status': r.status_code,
            'headers': r.headers,
            'body': r.content,
            'content_type': r.headers.get('Content-Type', ''),
            'size': len(r.content)
        })()
