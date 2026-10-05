<#
.SYNOPSIS
Run the Rust test suite at BelowNormal priority, with the test binaries running side by side.
By default a QUICK lane that leaves out the slowest tests listed in scripts\quick-skip.txt;
-Full runs everything, as CI always does.

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
separate build. A second run with nothing edited must print no `Compiling` line. Never set
`RUSTC_BOOTSTRAP` around this script: cargo treats it as a build input and rebuilds the graph.

Only artifacts cargo marks `profile.test` are run, so the viewer's `serve` binary and the
`verify_headless` example are built (the tests reach the first through cargo) but never launched.

THE QUICK LANE (the default). scripts\quick-skip.txt names whole tests and single frozen viewer
scenes that are left out of a local run because they are slow; CI calls `cargo test` itself and
never reads that file, so every one of them still runs before a push lands. A whole test is
skipped with `--exact --skip <name>` in its own binary only, and the run then checks that libtest
reports exactly that many filtered out — a stale entry or one that matched more than intended
fails the binary instead of passing quietly. A frozen scene is skipped through
PHYSSYNTH_FROZEN_SKIP, which frozen.rs validates (an entry naming no case panics). The summary
always says how much was skipped. Run -Full before a commit.

.PARAMETER Full
Run everything; ignore scripts\quick-skip.txt.

.PARAMETER Jobs
How many test binaries run at once.

.PARAMETER TestThreads
`--test-threads` for each binary. Jobs x TestThreads is the oversubscription.

.PARAMETER Filter
Regex over `<package>/<target>` (e.g. `physsynth-core/`, `frozen$`). Empty runs everything.

.PARAMETER NoDoc
Skip the doctest pass, which running binaries directly cannot cover.

.PARAMETER TestArgs
Forwarded to every binary, e.g. a test-name filter or `--nocapture`. Each element is split on
whitespace, because `powershell -File` delivers `-TestArgs a,b` as the ONE string "a,b": write
several as one string, `-TestArgs '--nocapture --ignored'`. Given a test-name filter, the
whole-test skips and the filtered-out count check are not applied (an `--exact` would turn the
filter into an exact match).

.EXAMPLE
powershell -File scripts\cargo-test-nice.ps1
powershell -File scripts\cargo-test-nice.ps1 -Full
powershell -File scripts\cargo-test-nice.ps1 -Filter 'physsynth-core/string_' -NoDoc
#>
param(
    [switch]$Full,
    [int]$Jobs = 8,
    [int]$TestThreads = 4,
    [string]$Filter = '',
    [switch]$NoDoc,
    [string[]]$TestArgs = @()
)

# Run as `.\cargo-test-nice.ps1` or `& …` this script shares the CALLER'S session, so everything
# it changes there is put back on every way out — above all the frozen skip list, which would
# otherwise make every later plain `cargo test` in that window skip scenes without a word.
$savedEnv = @{
    PHYSSYNTH_FROZEN_SKIP = $env:PHYSSYNTH_FROZEN_SKIP
    CARGO_MANIFEST_DIR    = $env:CARGO_MANIFEST_DIR
}
$savedPriority = [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass
$savedEap = $ErrorActionPreference

function Invoke-Main {
# Not 'Stop': under a host that captures stderr, PS 5.1 turns cargo's `Compiling` lines into
# terminating NativeCommandErrors. Every failure below is checked explicitly instead.
$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$skipFile = Join-Path $PSScriptRoot 'quick-skip.txt'
$work = 'W:\temp\claude\cargo-test-nice'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $work $stamp
$durationsFile = Join-Path $work 'durations.json'
New-Item -ItemType Directory -Force $logDir | Out-Null

$extra = @($TestArgs | ForEach-Object { $_ -split '\s+' } | Where-Object { $_ })
$userFilter = @($extra | Where-Object { -not $_.StartsWith('-') }).Count -gt 0

$self = [System.Diagnostics.Process]::GetCurrentProcess()
$self.PriorityClass = 'BelowNormal'
if ($self.PriorityClass -ne 'BelowNormal') { throw "priority did not drop: $($self.PriorityClass)" }
Write-Host "[cargo-test-nice] priority -> BelowNormal; logs -> $logDir"

# ---- the quick lane's list ----------------------------------------------------------------------
$skipTests = @{}      # '<package>/<target>' -> list of exact test names
$skipScenes = @()     # 'corpus/case'
if (-not $Full) {
    if (-not (Test-Path $skipFile)) { throw "quick lane needs $skipFile (or pass -Full)" }
    foreach ($raw in Get-Content $skipFile) {
        $line = ($raw -replace '#.*$', '').Trim()
        if (-not $line) { continue }
        $bin, $name = $line -split '\s*::\s*', 2
        if (-not $name) { throw "malformed line in ${skipFile}: '$raw'" }
        if ($bin -eq 'frozen') { $skipScenes += $name; continue }
        if (-not $skipTests.ContainsKey($bin)) { $skipTests[$bin] = @() }
        $skipTests[$bin] += $name
    }
    if ($userFilter) { $skipTests = @{} }
    $env:PHYSSYNTH_FROZEN_SKIP = $skipScenes -join ','
} else {
    Remove-Item Env:PHYSSYNTH_FROZEN_SKIP -ErrorAction SilentlyContinue
}

# ---- build once ---------------------------------------------------------------------------------
$manifest = Join-Path $repo 'Cargo.toml'
$json = & cargo test --workspace --release --no-run --manifest-path $manifest `
    --message-format=json-render-diagnostics
if ($LASTEXITCODE -ne 0) { Write-Host '[cargo-test-nice] build FAILED'; return $LASTEXITCODE }

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
$unknown = @($skipTests.Keys | Where-Object { $k = $_; -not ($bins | Where-Object Name -eq $k) })
if ($unknown.Count) { throw "$skipFile names test binaries that do not exist: $($unknown -join ', ')" }
if ($Filter) { $bins = @($bins | Where-Object { $_.Name -match $Filter }) }
if ($bins.Count -eq 0) { Write-Host '[cargo-test-nice] no test binary matched'; return 1 }

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
        $argv = @("--test-threads=$TestThreads") + $extra
        $skips = @()
        if ($skipTests.ContainsKey($b.Name)) { $skips = $skipTests[$b.Name] }
        if ($skips.Count) {
            $argv += '--exact'
            foreach ($s in $skips) { $argv += @('--skip', $s) }
        }
        $p = Start-Process -FilePath $b.Exe -ArgumentList $argv -WorkingDirectory $b.Dir `
            -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
        $null = $p.Handle  # PS 5.1: without caching the handle, ExitCode reads back as $null
        $running += [pscustomobject]@{
            Bin = $b; Proc = $p; Start = Get-Date; Out = $out; Skips = $skips.Count
        }
    }
    Start-Sleep -Milliseconds 200
    $still = @()
    foreach ($r in $running) {
        if (-not $r.Proc.HasExited) { $still += $r; continue }
        $r.Proc.WaitForExit()
        $secs = ((Get-Date) - $r.Start).TotalSeconds
        $text = Get-Content $r.Out -Raw
        $passed = 0; $failed = 0; $ignored = 0; $filtered = 0
        $pattern = 'test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored; ' +
            '\d+ measured; (\d+) filtered out'
        foreach ($mm in [regex]::Matches("$text", $pattern)) {
            $passed += [int]$mm.Groups[1].Value
            $failed += [int]$mm.Groups[2].Value
            $ignored += [int]$mm.Groups[3].Value
            $filtered += [int]$mm.Groups[4].Value
        }
        $code = $r.Proc.ExitCode
        $ok = ($code -eq 0)
        $note = ''
        # Every skip entry must remove exactly one test: fewer means a stale name, more means a
        # name that matched tests it was not meant to.
        if ($ok -and -not $userFilter -and $filtered -ne $r.Skips) {
            $ok = $false
            $note = "  ($filtered filtered out, $($r.Skips) skip entries)"
        }
        $tag = if ($ok) { 'ok  ' } else { 'FAIL' }
        $sk = if ($r.Skips) { " ($($r.Skips) skipped)" } else { '' }
        Write-Host ('[{0}] {1,7:N1}s  {2,4} passed  {3}{4}{5}' -f $tag, $secs, $passed,
            $r.Bin.Name, $sk, $note)
        $done += [pscustomobject]@{
            Name = $r.Bin.Name; Ok = $ok; Code = $code; Secs = $secs; Passed = $passed
            Failed = $failed; Ignored = $ignored; Filtered = $filtered; Log = $r.Out
        }
    }
    $running = $still
}
$wall = ((Get-Date) - $t0).TotalSeconds

$times = [ordered]@{}
foreach ($d in ($done | Sort-Object Name)) { $times[$d.Name] = [math]::Round($d.Secs, 2) }
# A filtered run refreshes only what it ran; the rest keep their last measurement.
foreach ($k in $known.Keys) { if (-not $times.Contains($k)) { $times[$k] = $known[$k] } }
$times | ConvertTo-Json | Set-Content -Encoding utf8 $durationsFile

# ---- doctests: running binaries directly cannot reach them --------------------------------------
$docOk = $true
if (-not $NoDoc -and -not $Filter) {
    Write-Host '[cargo-test-nice] doctests'
    & cargo test --workspace --release --doc --manifest-path $manifest | Out-Host
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
if ($Full) {
    Write-Host '[cargo-test-nice] FULL run: nothing skipped.'
} else {
    $nTests = ($skipTests.Values | ForEach-Object { $_.Count } | Measure-Object -Sum).Sum
    $msg = ("[cargo-test-nice] QUICK lane: skipped {0} slow tests and {1} frozen viewer scenes " +
        "listed in $skipFile. CI runs them all; use -Full before a commit.") -f
        [int]$nTests, $skipScenes.Count
    Write-Host $msg
}
foreach ($b in $bad) { Write-Host "  FAILED (exit $($b.Code)): $($b.Name)  log: $($b.Log)" }
if (-not $docOk) { Write-Host '  FAILED: doctests' }
if ($bad.Count -gt 0 -or -not $docOk) { return 1 }
return 0
}

$code = 1
try {
    # Everything the body writes to the pipeline is the exit code alone: output goes to the host.
    $code = Invoke-Main | Select-Object -Last 1
} finally {
    foreach ($k in $savedEnv.Keys) {
        if ($null -eq $savedEnv[$k]) {
            Remove-Item "Env:$k" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$k" $savedEnv[$k]
        }
    }
    [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = $savedPriority
    $ErrorActionPreference = $savedEap
}
exit $code
