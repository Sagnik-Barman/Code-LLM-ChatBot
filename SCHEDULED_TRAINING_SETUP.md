# Scheduled (not always-on) autonomous training

`autonomous_trainer_hf.py` run directly (`python autonomous_trainer_hf.py
--cycle_hours 6`) is a long-lived process with an internal `sleep()` timer --
if you close the terminal, sleep the PC, or shut down, that timer dies with
it. This setup instead uses Windows Task Scheduler to fire ONE training
cycle at a time (`--once`), with **"start when available"** turned on so a
missed cycle (PC was off) runs automatically as soon as you turn it back on,
rather than being silently skipped.

## Setup (run once)

1. Save `run_training_cycle.bat` to `D:\LLM-from-scratch\` (already
   configured to `cd` there and activate your venv -- if your project path
   differs, edit the `cd /d` line at the top).

2. Open **PowerShell as Administrator** and run:

```powershell
$action = New-ScheduledTaskAction -Execute "D:\LLM-from-scratch\run_training_cycle.bat"

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Hours 6) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -DontStopOnIdleEnd

Register-ScheduledTask -TaskName "CodeChatbotAutonomousTraining" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Runs one autonomous LoRA training cycle every 6 hours; catches up on cycles missed while the PC was off."
```

`-StartWhenAvailable` is the key setting for your situation: if the
scheduled time passes while the PC is off, Task Scheduler runs it as soon
as the PC is next on and idle-ish, instead of skipping it entirely.

## Checking on it

```powershell
Get-ScheduledTaskInfo -TaskName "CodeChatbotAutonomousTraining"   # last/next run time, last result code
Get-Content "D:\LLM-from-scratch\logs\autonomous_trainer_task.log" -Tail 50   # cycle output
```

Result code `0` means the last cycle ran without the script itself
crashing (it may still have skipped training if too few new examples were
filtered in -- check the log for that, not just the result code).

## Trigger a cycle manually, anytime

```powershell
Start-ScheduledTask -TaskName "CodeChatbotAutonomousTraining"
```

Useful right after a long chat session if you don't want to wait for the
next scheduled slot.

## Removing it later

```powershell
Unregister-ScheduledTask -TaskName "CodeChatbotAutonomousTraining" -Confirm:$false
```

## What this does and doesn't solve

- **Solves:** missed cycles while the PC is off get caught up automatically
  instead of silently vanishing.
- **Doesn't solve:** `app_hf.py` itself still isn't a background service --
  you still start it manually when you want to chat. That's fine: whenever
  you start it, it automatically resolves to whatever adapter was most
  recently promoted (via `get_live_adapter_path` in `code_model.py`), so
  you always get the latest trained version without tracking paths by hand.
- **If a cycle is still running when you shut down**, Windows will end the
  process (training isn't crash-resistant to a full power-off mid-run,
  only to a *script* crash). `--save_steps` checkpointing inside
  `lora_finetune.py`/`train_lora()` limits how much progress that costs you,
  but a hard shutdown mid-cycle isn't graceful. If you tend to shut down at
  predictable times, consider triggering a cycle manually
  (`Start-ScheduledTask`) before you do, rather than risking the clock
  running out mid-training.
