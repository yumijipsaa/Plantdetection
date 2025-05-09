# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .utils.env import setup_environment
from .config.defaults import _C
setup_environment()


# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.2.1"
