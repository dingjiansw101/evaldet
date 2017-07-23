import os
from GetFileFromDir import GetFileFromThisRootDir

def main():
    list = GetFileFromThisRootDir(r'E:\Data\dataset2\GFJLchips\GFJLtestchips\labelTxt', 'txt')
    outpath = r'E:\Data\dataset2\GFJLchips\GFJLtestchips\imagesets'
    setname = r'test.txt'
    outdir = os.path.join(outpath, setname)
    out = open(outdir, 'w')
    for dirname in list:
        basename = os.path.basename(dirname)
        filename = os.path.splitext(basename)[0]
        out.write(filename + '\n')
if __name__ == '__main__':
    main()