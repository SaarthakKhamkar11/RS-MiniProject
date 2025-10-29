"""
Startup script for Restaurant Recommendation System
Restaurant Recommendation System
"""

import subprocess
import sys
import os
import webbrowser
import time
import threading

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def start_server():
    """Start the Flask server"""
    print("Starting Restaurant Recommendation System server...")
    print("Restaurant Recommendation System")
    print("Hybrid Context-Aware Recommender System for Personalized Restaurant Suggestions")
    print("\n" + "="*80)
    print("Server will be available at: http://localhost:5000")
    print("Web interface will be available at: http://localhost:5000 (if configured)")
    print("="*80 + "\n")
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"✗ Error importing app: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"✗ Error starting server: {e}")

def open_browser():
    """Open browser after a delay"""
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass

def main():
    """Main function"""
    print("Restaurant Recommendation System")
    print("Restaurant Recommendation System")
    print("T.Y. B.Tech Sem-V | A.Y: 2025-26")
    print("\nInitializing system...")
    
    # Check if requirements are installed
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        print("✓ All required packages are available!")
    except ImportError:
        print("Installing required packages...")
        if not install_requirements():
            print("Failed to install requirements. Please install manually:")
            print("pip install -r requirements.txt")
            return
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()

