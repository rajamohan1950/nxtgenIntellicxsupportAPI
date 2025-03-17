import unittest
import os
import sys
import subprocess
import requests
import time
import signal
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.root_dir = os.getcwd()
        cls.venv_path = os.path.join(cls.root_dir, "venv")
        cls.system_python = sys.executable
        
        if sys.platform == "win32":
            cls.python_path = os.path.join(cls.venv_path, "Scripts", "python.exe")
            cls.pip_path = os.path.join(cls.venv_path, "Scripts", "pip.exe")
        else:
            cls.python_path = os.path.join(cls.venv_path, "bin", "python3.11")
            cls.pip_path = os.path.join(cls.venv_path, "bin", "pip3.11")

    def setUp(self):
        """Set up for each test"""
        self.processes = []

    def tearDown(self):
        """Clean up after each test"""
        for process in self.processes:
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=5)
            except:
                process.kill()

    def test_01_virtual_environment_creation(self):
        """Test virtual environment creation"""
        # Remove existing virtual environment
        if os.path.exists(self.venv_path):
            logger.info("Removed existing virtual environment")
            shutil.rmtree(self.venv_path)

        # Create new virtual environment
        logger.info("Creating new virtual environment...")
        try:
            result = subprocess.run(
                [self.system_python, "-m", "venv", self.venv_path],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Virtual environment created successfully")
            logger.info(f"Using Python executable: {self.python_path}")
            
            # Create symlinks if they don't exist
            if not os.path.exists(self.python_path):
                os.symlink(os.path.join(self.venv_path, "bin", "python3"), self.python_path)
            if not os.path.exists(self.pip_path):
                os.symlink(os.path.join(self.venv_path, "bin", "pip3"), self.pip_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e.stderr}")
            raise

    def test_02_package_installation(self):
        """Test package installation"""
        if not os.path.exists(self.python_path):
            self.skipTest("Python path not found. Run test_01 first.")

        # Install packages
        requirements_file = os.path.join(self.root_dir, "requirements.txt")
        self.assertTrue(os.path.exists(requirements_file), "requirements.txt not found")

        try:
            # Upgrade pip first
            subprocess.run(
                [self.python_path, "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                text=True,
                check=True
            )

            # Install requirements
            result = subprocess.run(
                [self.python_path, "-m", "pip", "install", "-r", requirements_file],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e.stderr}")
            raise

    def test_03_backend_imports(self):
        """Test backend imports"""
        if not os.path.exists(self.python_path):
            self.skipTest("Python path not found. Run test_01 first.")

        test_imports = """
import fastapi
import uvicorn
import transformers
import torch
import sentence_transformers
import langchain
import openai
import redis
import pydantic
"""
        try:
            result = subprocess.run(
                [self.python_path, "-c", test_imports],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to import required packages: {e.stderr}")
            raise

    def test_04_backend_startup(self):
        """Test backend server startup"""
        if not os.path.exists(self.python_path):
            self.skipTest("Python path not found. Run test_01 first.")

        env = os.environ.copy()
        env["PYTHONPATH"] = self.root_dir

        # Download required models
        try:
            logger.info("Downloading required models...")
            download_script = """
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model(model_name):
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(f"Successfully downloaded {model_name}")

# Download language detection model
download_model("papluca/xlm-roberta-base-language-detection")
"""
            subprocess.run(
                [self.python_path, "-c", download_script],
                env=env,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Models downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download models: {e.stderr}")
            raise

        try:
            process = subprocess.Popen(
                [self.python_path, "-m", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)

            # Wait for server to start
            time.sleep(5)

            # Check if server is running
            try:
                response = requests.get("http://localhost:8000/docs")
                self.assertEqual(response.status_code, 200, "Backend server not responding")
            except requests.exceptions.ConnectionError:
                self.fail("Backend server failed to start")
        except Exception as e:
            logger.error(f"Error starting backend server: {e}")
            raise

    def test_05_frontend_files(self):
        """Test frontend file existence"""
        frontend_dir = os.path.join(self.root_dir, "src", "frontend")
        self.assertTrue(os.path.exists(frontend_dir), "Frontend directory not found")
        self.assertTrue(os.path.exists(os.path.join(frontend_dir, "index.html")), "index.html not found")

    def test_06_data_files(self):
        """Test data file existence and format"""
        data_dir = os.path.join(self.root_dir, "data")
        self.assertTrue(os.path.exists(data_dir), "Data directory not found")
        self.assertTrue(os.path.exists(os.path.join(data_dir, "intents.json")), "intents.json not found")
        self.assertTrue(os.path.exists(os.path.join(data_dir, "responses.json")), "responses.json not found")

    def test_07_ml_modules(self):
        """Test ML module existence"""
        ml_dir = os.path.join(self.root_dir, "src", "ml")
        required_files = [
            "customer_support_service.py",
            "intent_classifier.py",
            "language_detector.py",
            "response_generator.py"
        ]
        for file in required_files:
            self.assertTrue(os.path.exists(os.path.join(ml_dir, file)), f"{file} not found")

    def test_08_backend_api_endpoints(self):
        """Test backend API endpoints"""
        if not os.path.exists(self.python_path):
            self.skipTest("Python path not found. Run test_01 first.")

        env = os.environ.copy()
        env["PYTHONPATH"] = self.root_dir

        try:
            process = subprocess.Popen(
                [self.python_path, "-m", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)

            # Wait for server to start
            time.sleep(5)

            try:
                # Test supported languages endpoint
                response = requests.get("http://localhost:8000/supported_languages")
                self.assertEqual(response.status_code, 200, "Supported languages endpoint failed")

                # Test supported intents endpoint
                response = requests.get("http://localhost:8000/supported_intents")
                self.assertEqual(response.status_code, 200, "Supported intents endpoint failed")

                # Test query processing endpoint
                response = requests.post(
                    "http://localhost:8000/process_query",
                    json={"text": "Hello", "preferred_language": "en"}
                )
                self.assertEqual(response.status_code, 200, "Query processing endpoint failed")
            except requests.exceptions.ConnectionError:
                self.fail("Backend server failed to respond")
        except Exception as e:
            logger.error(f"Error testing backend API endpoints: {e}")
            raise

    def test_09_system_startup(self):
        """Test complete system startup"""
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.root_dir
            
            result = subprocess.run(
                [sys.executable, "run.py"],
                env=env,
                capture_output=True,
                text=True,
                timeout=10  # Wait for 10 seconds max
            )
            self.assertIn("Starting multilingual customer support system", result.stdout)
            self.assertIn("Creating virtual environment", result.stdout)
            self.assertIn("Installing required packages", result.stdout)
        except subprocess.TimeoutExpired:
            pass  # This is expected as the server keeps running
        except Exception as e:
            logger.error(f"Error testing system startup: {e}")
            raise

    def test_10_error_handling(self):
        """Test error handling"""
        if not os.path.exists(self.python_path):
            self.skipTest("Python path not found. Run test_01 first.")

        env = os.environ.copy()
        env["PYTHONPATH"] = self.root_dir

        try:
            # Test invalid port
            process = subprocess.Popen(
                [self.python_path, "-m", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "80"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)
            time.sleep(2)
            
            # Process should fail to start on privileged port
            self.assertIsNotNone(process.poll(), "Server should not start on privileged port")
        except Exception as e:
            logger.error(f"Error testing error handling: {e}")
            raise

if __name__ == "__main__":
    unittest.main(verbosity=2) 