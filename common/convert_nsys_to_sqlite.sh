#!/bin/bash
# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

#
# Convert an NSYS .nsys-rep trace file to SQLite format.
#
# The single-trace analysis pipeline requires .sqlite or .json trace files.
# Use this script to convert .nsys-rep files before running the analysis.
#
# Usage:
#   ./convert_nsys_to_sqlite.sh <file.nsys-rep> [output.sqlite]
#
# If output path is omitted, the .sqlite file is written next to the input
# with the same base name (e.g., trace.nsys-rep -> trace.sqlite).
#
# Requires: nsys (NVIDIA Nsight Systems CLI) in PATH.

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <file.nsys-rep> [output.sqlite]"
    echo ""
    echo "Converts an NSYS .nsys-rep trace file to SQLite format."
    echo "The single-trace analysis pipeline only accepts .sqlite or .json files."
    exit 1
fi

INPUT="$1"

if [ ! -f "$INPUT" ]; then
    echo "Error: file not found: $INPUT"
    exit 1
fi

if [[ "$INPUT" != *.nsys-rep ]]; then
    echo "Error: expected a .nsys-rep file, got: $INPUT"
    exit 1
fi

if ! command -v nsys &>/dev/null; then
    echo "Error: nsys not found in PATH. Install NVIDIA Nsight Systems."
    exit 1
fi

if [ $# -eq 2 ]; then
    OUTPUT="$2"
else
    OUTPUT="${INPUT%.nsys-rep}.sqlite"
fi

echo "Converting: $INPUT -> $OUTPUT"
nsys export --type=sqlite --output="$OUTPUT" "$INPUT"
echo "Done: $OUTPUT"
