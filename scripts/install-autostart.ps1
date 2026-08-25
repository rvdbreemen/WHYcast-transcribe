<#
.SYNOPSIS
    Register (or remove) the WHYcast web UI server and worker as Windows
    scheduled tasks, so they come up with the machine.

.DESCRIPTION
    TASK-005 phase 4. Two scheduled tasks, one per process, both running as the
    current user at logon:

        WHYcast Web UI      python -m webui          (serves 127.0.0.1:8420)
        WHYcast Worker      python -m webui.worker   (runs one job at a time)

    They are separate on purpose. The worker is the thing that holds the GPU and
    that a CUDA crash can take down; the server surviving that is the point of
    ADR-008's process split, and two tasks preserve it - Windows restarts the
    one that died rather than both.

    RUN THIS YOURSELF. It changes machine configuration, so it is a script you
    read and execute, not something an assistant should do on your behalf. It
    needs no administrator rights: tasks are registered for the current user.

.PARAMETER Remove
    Unregister both tasks instead of creating them.

.PARAMETER Port
    Port for the web UI. Default 8420. The server binds 127.0.0.1 regardless
    (ADR-008 Decision Contract); this script never exposes it to the network.

.EXAMPLE
    .\scripts\install-autostart.ps1
    .\scripts\install-autostart.ps1 -Remove
#>

[CmdletBinding()]
param(
    [switch]$Remove,
    [int]$Port = 8420
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot 'venv\Scripts\pythonw.exe'
$ServerTask = 'WHYcast Web UI'
$WorkerTask = 'WHYcast Worker'

if ($Remove) {
    foreach ($name in @($ServerTask, $WorkerTask)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed scheduled task: $name"
        } else {
            Write-Host "Not registered, nothing to remove: $name"
        }
    }
    return
}

# pythonw.exe rather than python.exe: these run unattended, and a console
# window per process at every logon is a nuisance nobody asked for. Output goes
# to the log files below, and per-job logs live in logs/jobs/ regardless.
if (-not (Test-Path $Python)) {
    throw "No interpreter at $Python. Create the virtualenv first, or edit `$Python in this script."
}

$LogDir = Join-Path $RepoRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$tasks = @(
    @{ Name = $ServerTask
       Args = '-m webui'
       Desc = "WHYcast web UI on 127.0.0.1:$Port (ADR-008)." },
    @{ Name = $WorkerTask
       Args = '-m webui.worker'
       Desc = 'WHYcast job worker: one job at a time, holds the GPU (ADR-008).' }
)

foreach ($task in $tasks) {
    if (Get-ScheduledTask -TaskName $task.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $task.Name -Confirm:$false
        Write-Host "Replacing existing task: $($task.Name)"
    }

    $action = New-ScheduledTaskAction -Execute $Python -Argument $task.Args -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    # RestartCount/RestartInterval: a worker whose child took it down should come
    # back without someone logging in to notice. ExecutionTimeLimit 0 = never
    # kill it for running long; a full episode legitimately takes an hour.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

    Register-ScheduledTask `
        -TaskName $task.Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $task.Desc `
        -Force | Out-Null

    Write-Host "Registered: $($task.Name)  ->  $Python $($task.Args)"
}

Write-Host ''
Write-Host "Both tasks run at logon as $env:USERNAME."
Write-Host "Start them now without logging out:"
Write-Host "    Start-ScheduledTask -TaskName '$ServerTask'"
Write-Host "    Start-ScheduledTask -TaskName '$WorkerTask'"
Write-Host "Then open http://127.0.0.1:$Port"
Write-Host ''
Write-Host "Undo with: .\scripts\install-autostart.ps1 -Remove"
