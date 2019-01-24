@echo off
echo %VS90COMNTOOLS%
if "%platform%"=="x86" (
    call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x86
) else (
    call "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" %configuration% /x64
    call "%VS140COMNTOOLS%\..\..VC\vcvarsall.bat" x86_amd64
)
