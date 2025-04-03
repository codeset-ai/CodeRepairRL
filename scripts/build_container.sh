#!/bin/bash

echo "====================================================="
echo "  Building container... (this may take a while)"
echo "====================================================="
apptainer build crrl.sif scripts/container.def
