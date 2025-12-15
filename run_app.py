"""Launch the FashionMatch Streamlit app."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit app."""
    app_path = Path(__file__).parent / "src" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching FashionMatch UI...")
    print(f"📂 App path: {app_path}")
    print("\n" + "=" * 60)
    print("Access the app at: http://localhost:8501")
    print("=" * 60 + "\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
