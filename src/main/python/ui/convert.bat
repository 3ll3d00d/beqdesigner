for %%f in (*.ui) do (
    C:\Users\mattk\dev\conda\envs\beq\Scripts\pyuic5.exe %%f -o "%%~nf.py"
)
