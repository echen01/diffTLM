"""Bundled NVIDIA StyleGAN runtime; see LICENSE.txt for its license."""

import re


def namespace_imports(source):
    """Resolve checkpoint-embedded imports against the bundled runtime."""
    source = re.sub(
        r"^([ \t]*)from (torch_utils|dnnlib)(?=\.| import)",
        lambda match: f"{match[1]}from {__name__}.{match[2]}",
        source,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^([ \t]*)import (torch_utils|dnnlib)[ \t]*$",
        lambda match: f"{match[1]}from {__name__} import {match[2]}",
        source,
        flags=re.MULTILINE,
    )
