<#
.SYNOPSIS
Run the Rust test suite at BelowNormal priority, with the test binaries running side by side.

.DESCRIPTION
`cargo test` runs the threads inside one test binary in parallel but the ~90 binaries one after
another, so the slowest file sets the wall while most cores idle. This script builds every test
binary once, then runs them concurrently from a BelowNormal parent (Windows hands BelowNormal down
to every child: cargo, rustc and each test exe), longest-first by the durations of the last run.

Why not cargo-nextest: it runs every TEST as its own process, so a file whose tests share an
expensive fixture through a `OnceLock` (the whirl and phantom strings) would rebuild it per test.
Running per BINARY keeps that sharing.

No needless rebuilds: the build is always the one canonical invocation, `--workspace --release`,
which is CI's main job. Narrow a run with -Filter, which picks the binaries to RUN, never with
`-p`: a different package set resolves serde_json's `float_roundtrip` differently and that is a
separate build. A second run with nothing edited must print no `Compiling` line.

Only artifacts cargo marks `profile.test` are run, so the viewer's `serve` binary and the
`verify_headless` example are built (the tests reach the first through cargo) but never launched.

.PARAMETER Jobs
How many test binaries run at once.

.PARAMETER TestThreads
`--test-threads` for each binary. Jobs x TestThreads is the oversubscription.

.PARAMETER Filter
Regex over `<package>/<target>` (e.g. `physsynth-core/`, `frozen$`). Empty runs everything.

.PARAMETER NoDoc
Skip the doctest pass, which running binaries directly cannot cover.

.PARAMETER TestArgs
Forwarded to every binary after `--test-threads`, e.g. a test-name filter or `--ignored`.

.EXAMPLE
powershell -File scripts\cargo-test-nice.ps1
powershell -File scripts\cargo-test-nice.ps1 -Filter 'physsynth-core/string_' -NoDoc
#>
param(
    [int]$Jobs = 8,
    [int]$TestThreads = 4,
    [string]$Filter = '',
    [switch]$NoDoc,
    [string[]]$TestArgs = @()
)

# Not 'Stop': under a host that captures stderr, PS 5.1 turns cargo's `Compiling` lines into
# terminating NativeCommandErrors. Every failure below is checked explicitly instead.
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$work = 'W:\temp\claude\cargo-test-nice'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $work $stamp
$durationsFile = Join-Path $work 'durations.json'
New-Item -ItemType Directory -Force $logDir | Out-Null

$self = [System.Diagnostics.Process]::GetCurrentProcess()
$self.PriorityClass = 'BelowNormal'
if ($self.PriorityClass -ne 'BelowNormal') { throw "priority did not drop: $($self.PriorityClass)" }
Write-Host "[cargo-test-nice] priority -> BelowNormal; logs -> $logDir"

# ---- build once ---------------------------------------------------------------------------------
$manifest = Join-Path $repo 'Cargo.toml'
$json = & cargo test --workspace --release --no-run --manifest-path $manifest `
    --message-format=json-render-diagnostics
if ($LASTEXITCODE -ne 0) { Write-Host '[cargo-test-nice] build FAILED'; exit $LASTEXITCODE }

$bins = @()
foreach ($line in $json) {
    if (-not $line.StartsWith('{')) { continue }
    $m = $line | ConvertFrom-Json
    if ($m.reason -ne 'compiler-artifact' -or -not $m.profile.test -or -not $m.executable) {
        continue
    }
    $pkgDir = Split-Path -Parent $m.manifest_path
    $pkg = Split-Path -Leaf $pkgDir
    $bins += [pscustomobject]@{
        Name = "$pkg/$($m.target.name)"
        Exe  = $m.executable
        Dir  = $pkgDir
    }
}
if ($Filter) { $bins = @($bins | Where-Object { $_.Name -match $Filter }) }
if ($bins.Count -eq 0) { Write-Host '[cargo-test-nice] no test binary matched'; exit 1 }

# ---- longest first ------------------------------------------------------------------------------
$known = @{}
if (Test-Path $durationsFile) {
    $prev = Get-Content $durationsFile -Raw | ConvertFrom-Json
    foreach ($p in $prev.PSObject.Properties) { $known[$p.Name] = [double]$p.Value }
}
# A binary never timed sorts first: an unknown cost is safer started early than late.
$queue = New-Object System.Collections.Queue
$bins | Sort-Object { if ($known.ContainsKey($_.Name)) { $known[$_.Name] } else { 1e9 } } `
    -Descending | ForEach-Object { $queue.Enqueue($_) }

# ---- run ----------------------------------------------------------------------------------------
$running = @()
$done = @()
$t0 = Get-Date
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $Jobs) {
        $b = $queue.Dequeue()
        $safe = $b.Name -replace '[\\/:]', '__'
        $out = Join-Path $logDir "$safe.out.txt"
        $err = Join-Path $logDir "$safe.err.txt"
        # What `cargo test` sets at runtime; the tests read it at compile time today, but a
        # runtime read must not silently see the wrong crate.
        $env:CARGO_MANIFEST_DIR = $b.Dir
        $argv = @("--test-threads=$TestThreads") + $TestArgs
        $p = Start-Process -FilePath $b.Exe -ArgumentList $argv -WorkingDirectory $b.Dir `
            -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
        $null = $p.Handle  # PS 5.1: without caching the handle, ExitCode reads back as $null
        $running += [pscustomobject]@{ Bin = $b; Proc = $p; Start = Get-Date; Out = $out }
    }
    Start-Sleep -Milliseconds 200
    $still = @()
    foreach ($r in $running) {
        if (-not $r.Proc.HasExited) { $still += $r; continue }
        $r.Proc.WaitForExit()
        $secs = ((Get-Date) - $r.Start).TotalSeconds
        $text = Get-Content $r.Out -Raw
        $passed = 0; $failed = 0; $ignored = 0
        foreach ($mm in [regex]::Matches(
                "$text", 'test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored')) {
            $passed += [int]$mm.Groups[1].Value
            $failed += [int]$mm.Groups[2].Value
            $ignored += [int]$mm.Groups[3].Value
        }
        $code = $r.Proc.ExitCode
        $ok = ($code -eq 0)
        $tag = if ($ok) { 'ok  ' } else { 'FAIL' }
        Write-Host ('[{0}] {1,7:N1}s  {2,4} passed  {3}' -f $tag, $secs, $passed, $r.Bin.Name)
        $done += [pscustomobject]@{
            Name = $r.Bin.Name; Ok = $ok; Code = $code; Secs = $secs
            Passed = $passed; Failed = $failed; Ignored = $ignored; Log = $r.Out
        }
    }
    $running = $still
}
$wall = ((Get-Date) - $t0).TotalSeconds

$times = [ordered]@{}
foreach ($d in ($done | Sort-Object Name)) { $times[$d.Name] = [math]::Round($d.Secs, 2) }
if (-not $Filter) { $times | ConvertTo-Json | Set-Content -Encoding utf8 $durationsFile }
elseif (Test-Path $durationsFile) {
    # A filtered run refreshes only what it ran; the rest keep their last measurement.
    foreach ($k in $known.Keys) { if (-not $times.Contains($k)) { $times[$k] = $known[$k] } }
    $times | ConvertTo-Json | Set-Content -Encoding utf8 $durationsFile
}

# ---- doctests: running binaries directly cannot reach them --------------------------------------
$docOk = $true
if (-not $NoDoc -and -not $Filter) {
    Write-Host '[cargo-test-nice] doctests'
    & cargo test --workspace --release --doc --manifest-path $manifest
    $docOk = ($LASTEXITCODE -eq 0)
}

# ---- verdict ------------------------------------------------------------------------------------
$bad = @($done | Where-Object { -not $_.Ok })
$sumP = ($done | Measure-Object Passed -Sum).Sum
$sumF = ($done | Measure-Object Failed -Sum).Sum
$sumI = ($done | Measure-Object Ignored -Sum).Sum
Write-Host ''
Write-Host ('[cargo-test-nice] {0} binaries, {1} passed, {2} failed, {3} ignored, wall {4:N1}s' `
    -f $done.Count, $sumP, $sumF, $sumI, $wall)
foreach ($b in $bad) { Write-Host "  FAILED (exit $($b.Code)): $($b.Name)  log: $($b.Log)" }
if (-not $docOk) { Write-Host '  FAILED: doctests' }
if ($bad.Count -gt 0 -or -not $docOk) { exit 1 }
exit 0
