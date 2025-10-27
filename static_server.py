"""
Simple static file server for the frontend
Restaurant Recommendation System
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os

PORT = 8000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_static_server():
    """Start the static file server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Static server running at http://localhost:{PORT}")
        print("Serving frontend files...")
        httpd.serve_forever()

def open_browser():
    """Open browser after a delay"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{PORT}')

def main():
    """Main function"""
    print("Restaurant Recommendation System - Frontend Server")
    print("Restaurant Recommendation System")
    print("\nStarting static file server...")
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the static server
    start_static_server()

if __name__ == "__main__":
    main()

