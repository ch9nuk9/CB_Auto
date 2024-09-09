@echo off
REM Path to the MobaXterm executable
set MobaXtermPath="C:\Program Files (x86)\Mobatek\MobaXterm\MobaXterm.exe"

REM Predefined hostname
set hostname=login01.lisc.univie.ac.at

REM Prompt the user for their username
set /p username="Enter your username: "

REM Inform the user of the SSH connection details
echo Connecting to %hostname% as %username%...

REM Run MobaXterm with SSH login
%MobaXtermPath% -newtab "ssh %username%@%hostname%"

