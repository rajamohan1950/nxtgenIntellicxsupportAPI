import subprocess
import webbrowser
import time
import os
import signal
import sys
import threading
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_venv_activate_command():
    """Get the virtual environment activation command based on the platform."""
    if sys.platform == "win32":
        return os.path.join("venv", "Scripts", "activate.bat")
    return f"source {os.path.join('venv', 'bin', 'activate')}"

def ensure_venv():
    """Ensure virtual environment is created and packages are installed."""
    try:
        # Get the absolute path to the virtual environment
        venv_path = os.path.join(os.getcwd(), "venv")
        
        # Create or recreate virtual environment
        if os.path.exists(venv_path):
            logger.info("Removing existing virtual environment...")
            if sys.platform == "win32":
                subprocess.run(["rmdir", "/s", "/q", venv_path], check=True)
            else:
                subprocess.run(["rm", "-rf", venv_path], check=True)
        
        logger.info("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        
        # Get pip and python paths
        if sys.platform == "win32":
            pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
            python_path = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            # On Unix systems, check which Python executables exist
            bin_dir = os.path.join(venv_path, "bin")
            possible_python_paths = [
                os.path.join(bin_dir, "python"),
                os.path.join(bin_dir, "python3"),
                os.path.join(bin_dir, "python3.11")
            ]
            
            # Use the first Python executable that exists
            python_path = None
            for path in possible_python_paths:
                if os.path.exists(path):
                    python_path = path
                    break
            
            if not python_path:
                # If no Python executable found, use the one that created the venv
                python_path = os.path.join(bin_dir, "python3")
                if not os.path.exists(python_path):
                    # Create a symlink to the system Python
                    os.symlink(sys.executable, python_path)
            
            # Similarly for pip
            pip_path = os.path.join(bin_dir, "pip")
            if not os.path.exists(pip_path):
                pip_path = os.path.join(bin_dir, "pip3")
                if not os.path.exists(pip_path):
                    # Try to find pip in the bin directory
                    for file in os.listdir(bin_dir):
                        if file.startswith("pip"):
                            pip_path = os.path.join(bin_dir, file)
                            break
        
        logger.info(f"Using Python: {python_path}")
        logger.info(f"Using Pip: {pip_path}")
        
        logger.info("Installing required packages...")
        # Upgrade pip first
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        logger.info("Virtual environment setup completed successfully")
        
        # Return the python path for later use
        return python_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in ensure_venv: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ensure_venv: {str(e)}")
        raise

def stream_output(process, prefix):
    """Stream output from a process with a prefix"""
    def read_stream(stream, is_error=False):
        for line in stream:
            if isinstance(line, bytes):
                line = line.decode()
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Check if it's actually an error or just info
            if is_error:
                # Only log as error if it contains error keywords
                if any(keyword in line_stripped.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'failure']):
                    logger.error(f"{prefix} ERROR: {line_stripped}")
                else:
                    # Most stderr output from uvicorn is just INFO messages
                    logger.info(f"{prefix}: {line_stripped}")
            else:
                logger.info(f"{prefix}: {line_stripped}")
    
    # Start threads for stdout and stderr
    threading.Thread(target=read_stream, args=(process.stdout, False), daemon=True).start()
    threading.Thread(target=read_stream, args=(process.stderr, True), daemon=True).start()

def run_backend(python_path):
    """Run the FastAPI backend server."""
    try:
        # Create a new environment with the virtual environment activated
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()  # Add current directory to PYTHONPATH
        
        logger.info("Starting backend server with environment:")
        logger.info(f"PYTHONPATH: {env.get('PYTHONPATH')}")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Using Python: {python_path}")
        
        if not os.path.exists(python_path):
            raise FileNotFoundError(f"Python executable not found at {python_path}")
        
        cmd = [
            python_path,
            "-m", "uvicorn",
            "src.backend.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            text=True,
            env=env
        )
        
        # Start threads to stream output
        threading.Thread(target=stream_output, args=(process, "Backend"), daemon=True).start()
        
        return process
    except Exception as e:
        logger.error(f"Error in run_backend: {str(e)}")
        raise

def run_frontend():
    """Run a simple HTTP server for the frontend."""
    try:
        # Store the current working directory
        original_dir = os.getcwd()
        
        # Get the absolute path to the frontend directory
        frontend_dir = os.path.join(original_dir, "src", "frontend")
        
        # Create frontend directory if it doesn't exist
        os.makedirs(frontend_dir, exist_ok=True)
        
        # Change to the frontend directory
        os.chdir(frontend_dir)
        
        logger.info(f"Frontend server serving from: {os.getcwd()}")
        
        # Try ports starting from 8080
        port = 8080
        max_port = 8090  # Try up to port 8090
        httpd = None
        
        while port <= max_port:
            try:
                logger.info(f"Attempting to start frontend server on port {port}")
                httpd = HTTPServer(("localhost", port), SimpleHTTPRequestHandler)
                logger.info(f"Frontend server started on port {port}")
                break
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    logger.warning(f"Port {port} is already in use, trying next port")
                    port += 1
                else:
                    raise
        
        if httpd is None:
            raise RuntimeError(f"Could not find an available port between {8080} and {max_port}")
        
        # Return the server, original directory, and the port
        return httpd, original_dir, port
    except Exception as e:
        logger.error(f"Error in run_frontend: {str(e)}")
        raise

def main():
    try:
        logger.info("Starting multilingual customer support system...")
        
        # Ensure virtual environment is set up and get python path
        python_path = ensure_venv()
        
        # Start backend server
        logger.info("Starting backend server...")
        backend_process = run_backend(python_path)
        
        # Wait for backend to start and check if it's running
        logger.info("Waiting for backend server to start...")
        max_retries = 10
        retry_count = 0
        backend_ready = False
        
        while retry_count < max_retries and not backend_ready:
            try:
                # Try to connect to the backend health endpoint
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Backend server is running")
                    backend_ready = True
                else:
                    logger.warning(f"Backend server returned status code {response.status_code}")
                    time.sleep(2)
                    retry_count += 1
            except requests.exceptions.RequestException:
                logger.warning(f"Backend server not ready yet (attempt {retry_count+1}/{max_retries})")
                time.sleep(2)
                retry_count += 1
        
        if not backend_ready:
            logger.warning("Backend server may not be running properly, but continuing with frontend startup")
        
        # Start frontend server
        logger.info("Starting frontend server...")
        frontend_server, original_dir, port = run_frontend()
        
        # Open browser
        logger.info("Opening web browser...")
        webbrowser.open(f"http://localhost:{port}")
        
        try:
            # Keep the servers running
            frontend_server.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nShutting down servers...")
            # Stop backend process
            backend_process.send_signal(signal.SIGTERM)
            backend_process.wait()
            
            # Stop frontend server
            frontend_server.shutdown()
            frontend_server.server_close()
            
            # Change back to original directory
            os.chdir(original_dir)
            
            logger.info("Servers stopped successfully.")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()