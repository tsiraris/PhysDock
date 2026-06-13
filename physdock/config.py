"""
Configuration Loader Module for PhysDock.

This script acts as the foundational setup engine for the pipeline. It reads raw
YAML configuration files and processes them to support environment variable injection
(following the 12-factor application methodology). This means sensitive or machine-specific 
paths (like the installation directory of DiffDock) can be passed via `${ENV_VAR}` 
syntax rather than being hardcoded into the source code.

Finally, it wraps the parsed, expanded dictionary into a custom `Config` object that 
allows for clean, pythonic attribute access (e.g., `cfg.target.name` instead of 
`cfg['target']['name']`).
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([^}]+)\}")                                                     # Compiles a regular expression to match strings exactly formatted as ${VARIABLE_NAME}


def _expand(obj: Any) -> Any:
    """
    Recursively traverses a Python object and expands environment variables found in strings.

    It checks the type of the passed object. If it's a string, it uses the compiled
    regex to substitute `${VAR}` with the actual value from `os.environ`. If it's a
    dictionary or list, it recursively calls itself on every item/element. Unrecognized
    types (like ints or bools) are returned unmodified.

    Args:
        obj (Any): A string, list, dictionary, or primitive value loaded from the YAML.

    Returns:
        Any: The same object structure, but with all string environment variables expanded.

    Example:
        >>> os.environ['MY_DIR'] = '/usr/bin'
        >>> _expand({'path': '${MY_DIR}/app'})
        {'path': '/usr/bin/app'}
    """
    if isinstance(obj, str):                                                               # Checks if the current object being processed is a string type
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)          # Replaces ${VAR} with the system env value; keeps original string if VAR is missing
    if isinstance(obj, dict):                                                              # Checks if the current object being processed is a dictionary
        return {k: _expand(v) for k, v in obj.items()}                                     # Rebuilds the dictionary by recursively calling _expand on every dictionary value
    if isinstance(obj, list):                                                              # Checks if the current object being processed is a list
        return [_expand(v) for v in obj]                                                   # Rebuilds the list by recursively calling _expand on every item within the list
    return obj                                                                             # Returns the object unmodified if it is an integer, float, boolean, or None

# Sets up the class and forces to pass a standard Python dictionary (raw) when it is created.
@dataclass
class Config:
    """Thin attribute-access wrapper over the YAML so callers can do cfg.target.name instead 
    of cfg['target']['name'] while we keep the YAML as the single source of truth."""
    raw: dict

    def __getattr__(self, item: str) -> Any:
        """
        Enables dot-notation access to the underlying 'raw' dictionary keys.

        When Python intercepts an attribute access that doesn't exist on the object 
        (e.g., cfg.target), this method intercepts the call, looks up the key in the 
        raw dictionary, and wraps the result in a new Config object if it's a nested dict.

        Args:
            item (str): The name of the attribute/key being accessed by the user.

        Returns:
            Any: A new Config object if the retrieved value is a dict, else the raw value.

        Example:
            >>> cfg = Config({'app': {'port': 8080}})
            >>> cfg.app.port
            8080
        """
        try:                                                                               # Starts a try-except block to safely handle requests for keys that do not exist
            val = self.raw[item]                                                           # Attempts to retrieve the value from the raw dictionary using the requested attribute name
        except KeyError as e:                                                              # Catches the KeyError exception thrown if the dictionary does not contain the key
            raise AttributeError(item) from e                                              # Raises a standard AttributeError, preserving standard Python object behavior
        return Config(val) if isinstance(val, dict) else val                               # Wraps nested dicts in a new Config instance for chained dot-access; otherwise returns the primitive

    def get(self, *keys, default=None):
        """
        Safely retrieves deeply nested values using a sequence of keys, with a fallback default.

        It iterates through the provided keys sequentially, digging deeper into the raw nested
        dictionary at each step. If at any point a key is missing or the traversal hits a non-dictionary
        node, it immediately halts and returns the provided default value.

        Args:
            *keys (str): A variable number of string arguments representing the exact nested path.
            default (Any, optional): The value to return if the key path fails. Defaults to None.

        Returns:
            Any: The successfully retrieved value, or the default value if the path does not exist.

        Example:
            >>> cfg = Config({'app': {'port': 8080}})
            >>> cfg.get('app', 'host', default='localhost')
            'localhost'
        """
        node: Any = self.raw                                                               # Initializes a traversal pointer 'node', starting at the root raw dictionary
        for k in keys:                                                                     # Iterates sequentially over every key provided by the user in the arguments
            if not isinstance(node, dict) or k not in node:                                # Checks if the current node isn't a dictionary, or if the target key is missing
                return default                                                             # Immediately returns the fallback default value since the path is broken
            node = node[k]                                                                 # Updates the pointer to move one level deeper into the nested dictionary structure
        return node                                                                        # Returns the final, successfully retrieved value at the end of the key path


def load_config(path: str | Path) -> Config:
    """
    Reads a YAML file from disk, expands its environment variables, and returns a Config object.

    It opens the file specified by the file path in a context manager, parses it securely 
    using PyYAML's `safe_load` to prevent code execution vulnerabilities, passes the resulting 
    dictionary through the `_expand` function, and initializes a `Config` dataclass.

    Args:
        path (str | Path): The filesystem path to the target YAML configuration file.

    Returns:
        Config: A fully initialized Config object representing the expanded YAML data.

    Example:
        >>> cfg = load_config('config/target_config.yaml')
        >>> print(cfg.target.name)
    """
    with open(path) as fh:                                                                 # Opens the file specified by the path in read mode, ensuring safe file closure via context manager
        data = yaml.safe_load(fh)                                                          # Parses the text stream safely into a Python dictionary, preventing arbitrary code execution
    return Config(_expand(data))                                                           # Expands environment variables in the dict and wraps the final result in a Config object