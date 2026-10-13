# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Human-gated PR submission and AI-assisted review-fix loop.

Trust boundary: this package never talks to GitHub or to a git remote. Every
network operation lives in pr_flow/scripts/*.sh, which a human runs. The AI side
only reads fetched artifacts, edits files in a local worktree, and runs the GPU
verification harness.
"""
