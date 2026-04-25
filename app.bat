@echo off 

REM Set the path to your .config file
set CONFIG_PATH=.config

REM Read variables from the .config file
for /F "tokens=1,2 delims==" %%a in ('type "%CONFIG_PATH%"') do (
    if "%%a"=="DATA" set DATA=%%b
    if "%%a"=="EPOCHS" set EPOCHS=%%b
    if "%%a"=="LR0" set LR0=%%b
    if "%%a"=="PATIENCE" set PATIENCE=%%b
    if "%%a"=="MODEL" set MODEL=%%b
)

REM Define the output path for the best model
set BEST_MODEL_PATH=D:\PROJECTS\GODSEYE\weights\besttraining.pt

REM Execute the training command
echo Starting training...
yolo task=detect mode=train model="%MODEL%" data="%DATA%" epochs=%EPOCHS% lr0=%LR0% patience=%PATIENCE% save_period=1 project="D:\PROJECTS\GODSEYE\weights" name=besttraining

REM Copy the best model to the specified directory
echo Copying the best model to %BEST_MODEL_PATH%
copy "D:\PROJECTS\GODSEYE\weights\besttraining\weights\best.pt" "%BEST_MODEL_PATH%"

echo Training completed and best model saved to %BEST_MODEL_PATH%

pause
