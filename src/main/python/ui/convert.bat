for %%f in (*.ui) do (
    C:\Users\mattk\Anaconda3_64\envs\beq37\Scripts\pyside2-uic.exe %%f -o "%%~nf.py"
)

