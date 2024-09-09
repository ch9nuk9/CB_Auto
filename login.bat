@echo off
REM Predefined hostname
set hostname=login01.lisc.univie.ac.at

REM Prompt the user for their username
set /p username="Enter your username: "

REM Inform the user of the SSH connection details
echo Connecting to %hostname% as %username%...

REM Use the native SSH client for login
ssh %username%@%hostname%

REM Wait for the user to press a key before closing
pause
