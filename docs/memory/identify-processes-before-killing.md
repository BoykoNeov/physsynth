---
name: identify-processes-before-killing
description: "Never kill a process without first checking its command line — this machine runs the user's other Python work alongside mine; I killed two of their processes on 2026-07-17 by assuming they were my orphans"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 15ab45ea-6a58-479d-b986-a30b222ab481
---

Before terminating ANY process, check what it actually is — read its command line
(`Get-CimInstance Win32_Process | Select ProcessId, CommandLine`) and confirm it is mine.
Never infer ownership from the executable name, the start time, or the CPU burn.

**Why:** On 2026-07-17 I killed PIDs 16976 and 34036 believing they were pytest runs orphaned
by `TaskStop` (which kills the shell, not the child). They were the user's — confirmed by them.
This machine concurrently runs the user's own Python work (an `armor_pen` ballistics solver,
`rebake_all.py`). "python.exe with high CPU" is not evidence of my orphan; I expected exactly
one and found two, and that mismatch alone should have stopped me. A killed job is
unrecoverable — the user loses real compute, and I cannot undo it. Their instruction was
simply "be more careful next time."

**How to apply:** Identify first, kill second — always, even when I am confident and even when
a verifier is being starved by the contention. If the command line does not clearly show it is
mine (my pytest invocation, my server, my script), leave it alone and either wait or tell the
user what I see and let them decide. Prefer killing by a PID I recorded when *I* started the
process over discovering PIDs by scanning. Related: my own orphans are real — see
[[dont-run-pytest-suites-concurrently]] and the `TaskStop` gotcha in [[web-viewer-state]].
