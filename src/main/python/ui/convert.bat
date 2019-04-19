for %%f in (*.ui) do (
    C:\Users\mattk\Anaconda3_64\envs\beq37\Scripts\pyuic5.exe %%f -o "%%~nf.py"
)
