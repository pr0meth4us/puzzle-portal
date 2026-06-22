#!/usr/bin/env python3
import http.server
import socketserver
from pathlib import Path

PORT = 8089
FILE_PATH = Path("/tmp/local_clip.txt")

class ClipHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if FILE_PATH.exists():
            content = FILE_PATH.read_bytes()
        else:
            content = b"No clip data yet."
            
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)
        
    def log_message(self, format, *args):
        # Silence standard logs
        return

def main():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), ClipHandler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        pass

if __name__ == '__main__':
    main()
