"""
AegisVision AI — One-Click Launcher
Automatically starts the FastAPI server and opens your browser to the control room dashboard.
"""
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.5)
    print("\n[AegisVision AI] Opening Control Room Dashboard in your default browser...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 70)
    print("  🛡️  AEGIS VISION AI — ENTERPRISE VIDEO ANALYTICS PLATFORM")
    print("=" * 70)
    print("  • Dashboard URL  : http://localhost:8000")
    print("  • Documentation  : http://localhost:8000/docs")
    print("  • Camera / Source: Webcam 0 (auto-detects or falls back to test simulator)")
    print("=" * 70)
    
    # Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Uvicorn server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
