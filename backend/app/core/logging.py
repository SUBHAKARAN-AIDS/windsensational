# app/core/logging.py
#
# Central logging setup. Extracted VERBATIM from the top of main.py during
# restructuring (logging.basicConfig(level=logging.INFO)). Behavior unchanged.

import logging


def setup_logging():
    """Configure application logging. Idempotent enough for app startup."""
    logging.basicConfig(level=logging.INFO)
