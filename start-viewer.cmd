@echo off
rem Double-click to start the Physical Synthesis viewer.
rem
rem   1. If the viewer is already running on port 8000, just open it in the browser.
rem   2. Otherwise build it (cargo build --release; quick when nothing changed),
rem      start the server in this window, and open the browser once it answers.
rem
rem Close this window (or press Ctrl-C) to stop the viewer.
rem Pass "nobrowser" as the first argument to start the server without opening a browser.

setlocal
title Physical Synthesis viewer
cd /d "%~dp0"

set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%/"
set "EXE=%~dp0target\release\physsynth-viewer.exe"
set "STATIC=%~dp0web\static"

rem --- 1. Is something already answering on the port? -------------------------------------
rem exit 0 = it is this viewer, 2 = some other web server, 1 = nothing there
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%'; if ($r.Content -match 'Physical Synthesis') { exit 0 } else { exit 2 } } catch { if ($_.Exception.Response) { exit 2 } else { exit 1 } }"
if errorlevel 2 goto port_taken
if errorlevel 1 goto build
echo The viewer is already running at %URL%
echo Opening it in your browser. To run a newer build, close the viewer's window first.
if /i not "%~1"=="nobrowser" start "" "%URL%"
timeout /t 3 >nul 2>nul
exit /b 0

:port_taken
echo.
echo Port %PORT% is in use by another program, not this viewer.
echo Close that program, or start the viewer on another port by hand:
echo     cargo run --release -p physsynth-viewer -- serve --port 8001
echo.
pause
exit /b 1

rem --- 2. Build -----------------------------------------------------------------------------
:build
where cargo >nul 2>nul
if not errorlevel 1 goto do_build
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    goto do_build
)
if exist "%EXE%" goto no_cargo_but_built
echo.
echo Rust's cargo was not found, and the viewer has not been built yet.
echo Install Rust from https://rustup.rs and double-click this file again.
echo.
pause
exit /b 1

:no_cargo_but_built
echo cargo was not found - starting the viewer as it was last built.
goto run

:do_build
echo Building the viewer - the first build takes a few minutes, later ones a few seconds ...
cargo build --release -p physsynth-viewer
if not errorlevel 1 goto run
echo.
echo The build failed - see the messages above.
echo.
pause
exit /b 1

rem --- 3. Run -------------------------------------------------------------------------------
:run
if /i "%~1"=="nobrowser" goto serve
rem Open the browser in the background once the server answers (gives up after ~30 s).
start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%' | Out-Null; Start-Process '%URL%'; exit } catch { Start-Sleep -Milliseconds 500 } }"

:serve
echo.
echo Close this window to stop the viewer.
echo.
"%EXE%" serve --port %PORT% --static "%STATIC%"
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo The viewer stopped with an error - see the message above.
    echo.
    pause
)
exit /b %RC%
