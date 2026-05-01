import os, re

rootdir=('./')
for folder,dirs,file in os.walk(rootdir):
    for files in file:
        if files.endswith('.txt'):
            fullpath=open(os.path.join(folder,files),'r')
            for line in fullpath:
                if not re.fullmatch(r"^[a-zA-Z0-9(),+\-:/@' ]*$", line.strip()):
                    print(os.path.join(folder,files), line)
