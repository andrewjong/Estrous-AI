echo Copying experiments/ to models/ while keeping existing
cp -r -u -p experiments/* models/
echo Removing all but '*.pth' from models/
find models/ -type f -not -name '*.pth' -delete
echo Done
