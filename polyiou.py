import xml.etree.ElementTree as ET
import os
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapely.geometry as shgeo
from shapely.geometry import Polygon, MultiPoint

def calc_iou(poly1, poly2):
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    union_poly = poly1.union(poly2)
    union_area = union_poly.area
    iou = inter_area / union_area
    return iou

