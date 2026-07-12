@echo off
REM Runs ONE autonomous training cycle inside the project's venv, then exits.
REM Designed to be called repeatedly by Windows Task Scheduler rather than
REM run as a long-lived process -- see SCHEDULED_TRAINING_SETUP.md.

cd /d "D:\LLM-from-scratch"
call venv\Scripts\activate.bat

echo. >> logs\autonomous_trainer_task.log
echo ============================================== >> logs\autonomous_trainer_task.log
echo Cycle triggered at %date% %time% >> logs\autonomous_trainer_task.log

python autonomous_trainer_hf.py --once >> logs\autonomous_trainer_task.log 2>&1

echo Cycle finished at %date% %time% >> logs\autonomous_trainer_task.log
