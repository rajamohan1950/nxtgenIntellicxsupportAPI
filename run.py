import subprocess
import webbrowser
import time
import os
import signal
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

def run_backend():
    """Run the FastAPI backend server."""
    return subprocess.Popen(
        ["python", "src/backend/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def run_frontend():
    """Run a simple HTTP server for the frontend."""
    os.chdir("src/frontend")
    httpd = HTTPServer(("localhost", 8080), SimpleHTTPRequestHandler)
    return httpd

def main():
    print("Starting multilingual customer support system...")
    
    # Start backend server
    print("Starting backend server...")
    backend_process = run_backend()
    time.sleep(2)  # Wait for backend to start
    
    # Start frontend server
    print("Starting frontend server...")
    frontend_server = run_frontend()
    
    # Open browser
    print("Opening web browser...")
    webbrowser.open("http://localhost:8080")
    
    try:
        # Keep the servers running
        frontend_server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # Stop backend process
        backend_process.send_signal(signal.SIGTERM)
        backend_process.wait()
        
        # Stop frontend server
        frontend_server.shutdown()
        frontend_server.server_close()
        
        print("Servers stopped successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main() 