#!/bin/bash
# Optimize macOS for EHR AI Sidebar performance

echo "🚀 Optimizing macOS for EHR AI Sidebar..."

# Free up memory
echo "📦 Clearing caches..."
sudo purge

# Kill memory-intensive processes
echo "🔄 Stopping unnecessary processes..."
pkill -f "Google Chrome"
pkill -f "Slack"
pkill -f "Microsoft"

# Set Python memory limits
echo "🐍 Setting Python memory limits..."
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1

# Use lightweight Python packages
echo "📚 Installing optimized packages..."
pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 anthropic==0.7.8

echo "✅ Optimization complete!"
echo "💡 Recommended: Close other applications while running EHR AI"
echo "🎯 Use claude_only_main.py for best performance on your system"