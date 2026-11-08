@echo off
rem Builds Denoise.exe next to this folder with the C# compiler that ships with Windows (.NET Framework 4.x).
setlocal
set "CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" set "CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe"
"%CSC%" /nologo /codepage:65001 /target:winexe /optimize+ /win32icon:"%~dp0icon.ico" /r:System.Windows.Forms.dll /out:"%~dp0..\Denoise.exe" "%~dp0Denoise.cs"
