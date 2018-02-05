# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapely.geometry as shgeo
from shapely.geometry import Polygon, MultiPoint


def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    # polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def parse_gt(filename):
    objects = []
    with  open(filename, 'r', encoding='utf_16') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                object_struct['name'] = splitlines[8]
                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calc_iou(poly1, poly2):
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    union_poly = poly1.union(poly2)
    union_area = union_poly.area
    iou = inter_area / union_area
    return iou

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        #if i % 100 == 0:
         #   print ('Reading annotation for {:d}/{:d}'.format(
          #      i + 1, len(imagenames)) )
        # save
        #print ('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'w') as f:
         #   cPickle.dump(recs, f)
    #else:
        # load
        #with open(cachefile, 'r') as f:
         #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from comp4_det_test* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            def calcoverlaps(BBGT, bb):
                overlaps = []
                for index, GT in enumerate(BBGT):
                    # gtpoly = shgeo.Polygon([(BBGT[index, 0], BBGT[index, 1]),
                    #                         (BBGT[index, 2], BBGT[index,3]),
                    #                         (BBGT[index, 4], BBGT[index, 5]),
                    #                         (BBGT[index, 6], BBGT[index, 7])])
                    # detpoly = shgeo.Polygon([(bb[0], bb[1]),
                    #                          (bb[2], bb[3]),
                    #                          (bb[4], bb[5]),
                    #                          (bb[6], bb[7])])
                    # overlap = calc_iou(gtpoly, detpoly)
                    overlap = polygon_iou(BBGT[index], bb)
                    overlaps.append(overlap)
                return overlaps
            overlaps = calcoverlaps(BBGT, bb)

            # ixmin = np.maximum(BBGT[:, 0], bb[0])
            # iymin = np.maximum(BBGT[:, 1], bb[1])
            # ixmax = np.minimum(BBGT[:, 2], bb[2])
            # iymax = np.minimum(BBGT[:, 3], bb[3])
            # iw = np.maximum(ixmax - ixmin + 1., 0.)
            # ih = np.maximum(iymax - iymin + 1., 0.)
            # inters = iw * ih
            #
            # # union
            # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            #
            # overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    print('check fp:', fp)
    print('check tp', tp)


    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    #print('fp num:', fp)
    #print('tp num:', tp)
    #print('np num:', np)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

# datamap2 = {'0A': 'passenger plane', '0B': 'fighter aeroplane', '0C': 'radar',
#            '1': 'baseball diamond', '2': 'bridge', '3': 'ground track', '4A': 'car', '4B': 'trunck',
#            '4C': 'bus', '5A': 'ship','5B': 'big ship', '6': 'tennis court', '7': 'baseketball court',
#            '8': 'storage tank', '9': 'soccer ball field', '10': 'turntable',
#            '11': 'harbor', '12': 'electric pole', '13': 'parking lot', '14': 'swimming pool', '15': 'lake',
#            '16': 'helicopter', '17': 'airport', '18A': 'viaduct'}
datamap2 = {'0A': 'plane', '0B': 'fighter aeroplane', '0C': 'radar',
           '1': 'baseball-diamond', '2': 'bridge', '3': 'ground-track', '4A': 'car', '4B': 'trunck',
           '4C': 'bus', '5A': 'ship','5B': 'big ship', '6': 'tennis court', '7': 'baseketball-court',
           '8': 'storage-tank', '9': 'soccer-ball-field', '10': 'turntable',
           '11': 'harbor', '12': 'electric-pole', '13': 'parking-lot', '14': 'swimming-pool', '15': 'lake',
           '16': 'helicopter', '17': 'airport', '18A': 'viaduct'}
def main():
    detpath = r'E:\bod-dataset\results\bod_ssd1024_2000000-nms_dots8\comp4_det_test_{:s}.txt'
    #annopath = r'G:\voc_eval\GFformatPascal\{:s}.txt'
    annopath = r'E:\bod-dataset\wordlabelBestStart\{:s}.txt'
    #annopath = r'E:\bod-dataset\testset\ReclabelTxt\{:s}.txt'
    imagesetfile = r'E:\bod-dataset\testset\testset.txt'
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'turntable', 'harbor', 'swimming-pool', 'helicopter']
    #classnames = ['ground-track-field']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)
        plt.figure(figsize=(8,4))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec)
       # plt.show()
    map = map/len(classnames)
    print('map:', map)
    #classout = ' && '.join(classes)
    #print('classout: ', classout)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)
    #print('aps:', ' '.join(map(str, classaps)))
if __name__ == '__main__':
    main()