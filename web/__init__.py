"""Web-viewer wrapper (Phase 3.5).

A *wrapper* layer (HANDOFF §3.1): a local HTTP backend + static frontend that depend on the headless
DSP core, never the reverse. No physics lives here — only serialization and transport. The core
(``physsynth/core``) is untouched and stays importable, pure, and testable on its own.
"""
