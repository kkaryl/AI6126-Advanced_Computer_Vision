@echo on
REM
setlocal
cd /D "%~dp0"
call %UserProfile%\anaconda3\Scripts\activate.bat
REM
call conda activate ai6126acv_p1
REM
call jupyter-lab
