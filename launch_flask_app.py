#!/usr/bin/env python3
"""
Launch the Flask Exoplanet Detection Web Application
Web interface for the hierarchical ensemble model system
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'tensorflow',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"Installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}")
                return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        'robust_xgboost_model.pkl',
        'robust_random_forest_model.pkl',
        'robust_hierarchical_ensemble.h5',
        'robust_scaler.pkl',
        'robust_feature_columns.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing model files: {missing_files}")
        print("Please run model training first:")
        print("   python robust_hierarchical_training.py")
        return False
    
    return True

def launch_flask_app():
    """Launch the Flask web application"""
    print("Starting Professional Exoplanet Detection Platform...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install required dependencies")
        return False
    
    # Check model files
    if not check_model_files():
        print("Missing required model files")
        return False
    
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    try:
        print("Starting Flask web application...")
        print("The web app will open in your default browser")
        print("If it doesn't open automatically, go to: http://localhost:8080")
        print("\n" + "="*60)
        print("Professional Exoplanet Detection Platform is now running!")
        print("="*60)
        print("\nFeatures:")
        print("   • Modern, minimalist design")
        print("   • Single exoplanet prediction")
        print("   • Batch analysis with CSV upload")
        print("   • Comprehensive model statistics")
        print("   • Educational content about exoplanets")
        print("   • Hierarchical ensemble architecture")
        print("\nPerfect for hackathon presentations!")
        print("="*60)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:8080')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Launch the Flask app
        subprocess.run([
            sys.executable, 'flask_app.py'
        ])
        
    except KeyboardInterrupt:
        print("\nWeb application stopped by user")
    except Exception as e:
        print(f"Error launching web application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("Exoplanet Detection Platform Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('flask_app.py'):
        print("flask_app.py not found in current directory")
        print("Please run this script from the project root directory")
        return
    
    # Launch the Flask webapp
    success = launch_flask_app()
    
    if success:
        print("Web application launched successfully!")
    else:
        print("Failed to launch web application")

if __name__ == "__main__":
    main()
