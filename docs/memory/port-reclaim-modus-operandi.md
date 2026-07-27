---
name: port-reclaim-modus-operandi
description: "Before binding a port, check whether a stale run of THIS program holds it — kill exactly that PID and reclaim; else try the next port"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 85bdbf26-99a4-45de-b372-8540c65a1f65
  modified: 2026-07-20T07:48:17.564Z
---

When I need a port (e.g. `python web/server.py --port 8000` for the viewer), do NOT
blindly bind and do NOT blindly pick a random free port. The standing procedure:

1. Check whether the intended port is already listening.
2. If it is, identify the owning PID **and read its full command line**.
3. If — and only if — it is a previous session of *this same program* (the viewer
   server / the thing I started earlier in this project), kill exactly that PID and
   take the port back.
4. If it belongs to anything else, leave it alone and try the next port
   (8001, 8002, …) instead.

**Why:** A stale server from an earlier session is mine to reclaim, and reclaiming the
canonical port keeps URLs stable across sessions. But this machine runs the user's own
processes — see [[identify-processes-before-killing]] — so "port is busy" is never
sufficient justification to kill. Identity of the process is the gate, not occupancy
of the port.

**How to apply:** On Windows/PowerShell,
`Get-NetTCPConnection -LocalPort 8000 -State Listen` → `OwningProcess`, then
`Get-CimInstance Win32_Process -Filter "ProcessId=<pid>" | Select CommandLine` to
confirm it is the project's own server before `Stop-Process`. Relevant to
[[web-viewer-state]] (default port 8000).
