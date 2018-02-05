import os
import codecs

def main():
    path = r'E:\yanshen\dataset2\trainval\results'
    imgname = r'JL101A_PMS_20160515010800_000008725_201_0013_001_L1_PAN_1_2'
    dir1 = os.path.join(path, imgname + '.txt')
    f = open(dir1, 'r', encoding='utf_16')
    outdir = r'E:\yanshen\dataset2\trainval\results\comp4_det_test_aeroplane.txt'
    f_out = open(outdir, 'w')
    #f_out = codecs.open(outdir, 'w', 'utf_16')
    while True:
        line = f.readline()
        if line:
            splitline = line.strip().split(' ')
            outlist = [imgname]
            outlist.append(str(float(0.8)))
            outlist.append(splitline[0])
            outlist.append(splitline[1])
            outlist.append(splitline[4])
            outlist.append(splitline[5])
            outline = ' '.join(outlist)
            f_out.write(outline + '\n')
        else:
            break
if __name__ == '__main__':
    main()