# Wait for the judge pipeline to finish, preserve its logs, then shut down.
#
# Tied to judge_run.py rather than train.py on purpose: shutting down when
# training ends would kill the machine before anything had been evaluated, so
# the run would finish with no answer about whether the model improved.
#
# Cancel at any point with:  shutdown /a

$ErrorActionPreference = 'SilentlyContinue'

$repo  = "C:\Users\Pranesh A S\Documents\semicon-rtx-port\semicon"
$tasks = "C:\Users\PRANES~1\AppData\Local\Temp\claude\C--Users-Pranesh-A-S-Documents-semicon-rtx-port\4c828275-eaa2-42f7-b395-9d76f13ac4af\tasks"
$logs  = Join-Path $repo "logs\rtx_session"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Judging {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -like '*judge_run.py*' }).Count -gt 0
}

# Guard against firing instantly if this starts before judge_run.py is up.
$waited = 0
while (-not (Judging) -and $waited -lt 300) { Start-Sleep -Seconds 10; $waited += 10 }

while (Judging) { Start-Sleep -Seconds 30 }

# The task logs live under the session temp directory, which does not survive.
# Copy everything needed to read the result after the machine comes back.
Copy-Item "$tasks\blgnspjyl.output" (Join-Path $logs "train_v5f.log")     -Force
Copy-Item "$tasks\b44r38v59.output" (Join-Path $logs "judge_v5f.log")     -Force
Copy-Item "$tasks\b4k2ncote.output" (Join-Path $logs "build_pool.log")    -Force
Copy-Item "$repo\weights\driftsense_v5f_history.json" $logs              -Force
Copy-Item "$repo\results\stream\*.json" $logs                            -Force

@(
    "Drift-Sense RTX session"
    "finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    ""
    "Read these:"
    "  logs\rtx_session\judge_v5f.log   <- the verdict: stream_eval + paired bootstrap"
    "  logs\rtx_session\train_v5f.log   <- per-epoch training history"
    "  weights\driftsense_v5f_e*.pt     <- every epoch kept"
    "  weights\driftsense_v5f_swa.pt    <- averaged tail checkpoints"
    ""
    "weights\driftsense.pt was never written. The submission is intact."
) | Out-File (Join-Path $logs "READ_ME_FIRST.txt") -Encoding utf8

shutdown /s /t 300 /c "Drift-Sense run complete. Shutting down in 5 minutes - run 'shutdown /a' to cancel."
