# Copyright (c) 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Marker for the LLM use-case subtree.

Everything GPU-inference-framework specific lives here: profiling
(auto_profile, auto_profile_server_client), trace analysis (auto_analyze),
code generation (auto_code_gen), model inventory (auto_config), and the
campaign run configs (configs). Use-case-neutral code lives in common/;
the bug-fix backport pipelines (auto_bug_fix, backport_scanner) are
separate and stay at the repository top level.
"""
