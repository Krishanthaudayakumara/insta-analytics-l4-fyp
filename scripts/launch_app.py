#!/usr/bin/env python3
"""
Instagram Engagement Prediction System - App Launcher
Easy launcher for both application versions
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch Instagram Engagement Prediction System')
    parser.add_argument(
        '--app', 
        choices=['main', 'legacy'], 
        default='main',
        help='Choose which app to launch: main (modular) or legacy (enhanced monolithic)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port to run the Streamlit app on (default: 8501)'
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Choose app file
    if args.app == 'main':
        app_file = 'app.py'
        print("🚀 Launching Main Modular Application (Clean Architecture)...")
    else:
        app_file = 'app_legacy.py'
        print("🏗️ Launching Legacy Enhanced Application (Monolithic)...")
    
    # Launch command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', app_file,
        '--server.port', str(args.port),
        '--server.headless', 'false'
    ]
    
    print(f"📸 Starting Instagram Engagement Prediction System...")
    print(f"🌐 Application will be available at: http://localhost:{args.port}")
    print(f"📁 Working directory: {project_root}")
    print(f"⚙️ Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
