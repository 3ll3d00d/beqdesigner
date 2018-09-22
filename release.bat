set release=%1
echo "Releasing %release%"
git semver --next-%release% > src\main\python\VERSION
if errorlevel 1 exit
set /p VERSION=<src\main\python\VERSION
echo "Next version is %VERSION%"
git add src\main\python\VERSION
git commit -m"release: %VERSION%"
git tag -am %VERSION% %VERSION%

echo "Creating exe"
pyinstaller --clean --log-level=ERROR -F for_exe.spec
echo "Creating installer content"
pyinstaller --clean --log-level=WARN -D for_nsis.spec
echo "Creating installer"
"C:\Program Files (x86)\NSIS\makensis" src\main\nsis\Installer.nsi

echo " ***** READY TO PUBLISH ***** "