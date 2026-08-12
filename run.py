#!/usr/bin/env python3
"""Start YAAP:  python run.py  →  http://127.0.0.1:8811"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8811, reload=False)
