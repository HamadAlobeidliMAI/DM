@echo off
REM Run one dataset with Ollama (change DATA to try others)
set DATA=massive_scenario
set MODEL=qwen2.5:7b
if not exist logs mkdir logs
if not exist generated_labels mkdir generated_labels
python select_part_labels.py
python label_generation.py --data %DATA% --model %MODEL%
python given_label_classification.py --data %DATA% --model %MODEL%
python evaluate.py --data %DATA% --predict_file %DATA%_small_find_labels.json
pause
