import subprocess
import sys
import os

script_path = os.path.join(os.path.dirname(__file__), "app_streamlit.py")
print(script_path)
subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
