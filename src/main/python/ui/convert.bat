for %%f in (*.ui) do (
    C:\Users\mattk\dev\conda\envs\beq\Scripts\pyside2-uic.exe %%f -o "%%~nf.py"
)
