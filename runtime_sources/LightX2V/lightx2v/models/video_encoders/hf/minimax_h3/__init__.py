"""Minimal package boundary used by X-MinimaxH3's audio-VAE loader.

The release vendors ``weights.py`` because the native audio VAE shares its
safetensors subset loader.  Importing the full LightX2V video VAE here would
pull in an unrelated runtime graph and dependencies that the service does not
use, so the package deliberately has no eager re-exports.
"""
