@echo off
echo ==============================================
echo Installing Windows Dependencies...
echo ==============================================
pip install -r requirements_windows.txt

echo.
echo ==============================================
echo Building AuraController Windows Executable (GUI)
echo ==============================================
REM Using --onedir instead of --onefile is extremely important for OpenCV/MediaPipe!
REM Standard --onefile extract takes >10 seconds every time you double click because of AI models!

pyinstaller --noconfirm --onedir --windowed --add-data "hand_landmarker.task;." --add-data "face_landmarker.task;." AuraController.py

echo.
echo ==============================================
echo Build Complete! 
echo Check the 'dist/AuraController' folder for your executable.
echo You can run AuraController.exe without Python installed anymore!
echo ==============================================
pause
