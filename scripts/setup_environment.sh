#!/bin/bash
# scripts/setup_dev_env.sh

echo "Setting up GraphCast Mars development environment..."

# Create external directory
mkdir -p external

# Clone GraphCast if not exists
if [ ! -d "external/graphcast" ]; then
    echo "Cloning GraphCast..."
    git clone https://github.com/google-deepmind/graphcast.git external/graphcast
else
    echo "GraphCast already cloned"
fi

# Add to Python path using .pth file
# SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

# if [ -n "$SITE_PACKAGES" ]; then
#     PROJECT_ROOT=${PWD}
#     echo "${PROJECT_ROOT}/external/graphcast" > "${SITE_PACKAGES}/graphcast_dev.pth"
#     echo "✅ Added GraphCast to Python path: ${SITE_PACKAGES}/graphcast_dev.pth"
# else
#ß echo "⚠️  Could not find site-packages, using PYTHONPATH instead"
export PYTHONPATH="${PYTHONPATH}:${PWD}/external/graphcast"
echo "export PYTHONPATH=\"\${PYTHONPATH}:${PWD}/external/graphcast\"" >> .env
echo "✅ Added to PYTHONPATH (saved to .env)"
# fi

# Add MCD library to Python path
PROJECT_ROOT=${PWD}
cp /discover/nobackup/projects/QEFM/data/shared/mcd_lib/*.so ${PROJECT_ROOT}/src/preprocessing/.
MCD_LIB_PATH="${PPROJECT_ROOT}/src/preprocessing/"
export PYTHONPATH="${PYTHONPATH}:${MCD_LIB_PATH}"
echo "export PYTHONPATH=\"\${PYTHONPATH}:${MCD_LIB_PATH}\"" >> .env
echo "✅ Added to PYTHONPATH (saved to .env)"

# Install other dependencies
#echo "📦 Installing dependencies..."
#pip install -r requirements.txt

# Install project in development mode
#echo "📦 Installing graphcast-mars in development mode..."
#pip install -e .

echo ""
echo "✅ Development environment ready!"
echo ""
# echo "To activate in new terminal sessions:"
# echo "  source .env  # If using PYTHONPATH method"
# echo ""
# echo "Quick start:"
# echo "  python train.py --config configs/training_config.yaml"