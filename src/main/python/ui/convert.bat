for %%f in (*.ui) do (
    C:\Users\mattk\Anaconda3\envs\scanner32\Library\bin\pyuic5.bat %%f -o "%%~nf.py"
)
