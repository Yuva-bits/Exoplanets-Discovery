#!/usr/bin/env python3
"""
Restart script to demonstrate automatic model building on startup
"""

import subprocess
import time
import sys
import os

def kill_existing_flask():
    """Kill any existing Flask processes"""
    try:
        subprocess.run(['pkill', '-f', 'flask_app.py'], check=False)
        print("Stopping existing Flask app...")
        time.sleep(2)
    except Exception as e:
        print(f"Note: {e}")

def start_flask_with_building():
    """Start Flask app which will automatically build models"""
    print("Starting Flask app with automatic model building...")
    print("=" * 60)
    print("The app will now:")
    print("   1. Load data from CSV files")
    print("   2. Build XGBoost model")
    print("   3. Build Random Forest model") 
    print("   4. Build Hierarchical Neural Network")
    print("   5. Save all models")
    print("   6. Start web server")
    print("=" * 60)
    print()
    
    try:
        # Change to the correct directory
        os.chdir('/Users/yuvashreesenthilmurugan/Documents/Projects/NasaSpaceApps2/NasaSpaceApps2')
        
        # Start Flask app
        process = subprocess.Popen([sys.executable, 'flask_app.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 universal_newlines=True)
        
        # Stream output to show training progress
        for line in process.stdout:
            print(line.rstrip())
            
    except KeyboardInterrupt:
        print("\nStopping Flask app...")
        process.terminate()
    except Exception as e:
        print(f"Error starting Flask app: {e}")

if __name__ == "__main__":
    print("Exoplanet Detection Platform - Restart with Building")
    print("=" * 60)
    
    kill_existing_flask()
    start_flask_with_building()
