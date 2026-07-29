#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Aegean Vessel Tracker..."
echo ""
python3 setup.py
read -p "Press Enter to close..."
