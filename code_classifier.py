from collections import namedtuple
import copy
from functools import partial
import json
import os
import pickle
from xml.dom import minidom

import cv2
#import ipyleaflet as ipyl
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from shapely.geometry import mapping, shape, Polygon
from shapely.ops import transform
from sklearn import neighbors
from sklearn.metrics import classification_report





train_scene_filename = '../input/finaldata/test_scene_cropped.tif'
train_metadata_filename = '../input/finaldata/test_scene_cropped_metadata.xml'
train_ground_truth_filename = '../input/finaldata/ground-truth-test.geojson'






test_scene_filename = '../input/finaldata/train_scene_cropped.tif'
test_metadata_filename = '../input/finaldata/train_scene_cropped_metadata.xml'
test_ground_truth_filename = '../input/finaldata/ground-truth-train.geojson'





def project_feature(feature, proj_fcn):
    g1 = shape(feature['geometry'])
    g2 = transform(proj_fcn, g1)
    proj_feat = copy.deepcopy(feature)
    proj_feat['geometry'] = mapping(g2)
    return proj_feat

#Referenced from planet.com
def project_features_to_srs(features, img_srs, src_srs='epsg:4326'):
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init=src_srs),
        pyproj.Proj(init=img_srs))
    
    return [project_feature(f, proj_fcn) for f in features]


def polygon_to_contour(feature_geometry, image_transform):
    points_xy = shape(feature_geometry).exterior.coords
    points_x, points_y = zip(*points_xy) # inverse of zip
    rows, cols = rasterio.transform.rowcol(image_transform, points_x, points_y)
    return np.array([pnt for pnt in zip(cols, rows)], dtype=np.int32)





def load_geojson(filename):
    with open(filename, 'r') as f:
        return json.load(f)



def load_contours(ground_truth_filename, pl_filename):
    with rasterio.open(pl_filename) as img:
        img_transform = img.transform
        img_srs = img.crs['init']

    ground_truth_data = load_geojson(ground_truth_filename)#test

    # project to image srs
    projected_features = project_features_to_srs(ground_truth_data, img_srs)#shape to srs

    # convert projected features to contours
    contours = [polygon_to_contour(f['geometry'], img_transform)
                for f in projected_features]
    return contours

#print(len(load_contours(train_ground_truth_filename, train_scene_filename)))



#Referenced
NamedBands = namedtuple('NamedBands', 'b, g, r, nir')

#Not this one
def load_bands(filename):
    with rasterio.open(filename) as src:
        b, g, r, nir = src.read()
        mask = src.read_masks(1) == 0 # 0 value means the pixel is masked
    
    bands = NamedBands(b=b, g=g, r=r, nir=nir)
    return NamedBands(*[np.ma.array(b, mask=mask)
                        for b in bands])

def get_rgb(named_bands):
    return [named_bands.r, named_bands.g, named_bands.b]

def check_mask(band):
    return '{}/{} masked'.format(band.mask.sum(), band.mask.size)
    
#print(check_mask(load_bands(train_scene_filename).r))



NamedCoefficients = namedtuple('NamedCoefficients', 'b, g, r, nir')

def read_refl_coefficients(metadata_filename):
    xmldoc = minidom.parse(metadata_filename)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    return NamedCoefficients(b=coeffs[1],
                             g=coeffs[2],
                             r=coeffs[3],
                             nir=coeffs[4])



def load_refl_bands(filename, metadata_filename):
    bands = load_bands(filename)
    coeffs = read_refl_coefficients(metadata_filename)
    return NamedBands(*[b.astype(float)*c for b,c in zip(bands, coeffs)])


#print(read_refl_coefficients(train_metadata_filename))



#NDVI
def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
    return (ndarray - old_min)*(new_max - new_min)/(old_max - old_min) + new_min

def _mask_to_alpha(bands):
    band = np.atleast_3d(bands[0])
    alpha = np.zeros_like(band)
    alpha[~band.mask] = 1
    return alpha

def _add_alpha_mask(bands):
    return np.dstack([bands, _mask_to_alpha(bands)])

#Referenced from planet.com
def bands_to_display(bands, alpha=False):
#     """Converts a list of 3 bands to a 3-band rgb, normalized array for display."""  
    assert len(bands) in [1,3]
    all_bands = np.dstack(bands)
    old_min = np.percentile(all_bands, 2)
    old_max = np.percentile(all_bands, 98)

    new_min = 0
    new_max = 1
    scaled = [np.clip(_linear_scale(b.astype(np.double),
                                    old_min, old_max, new_min, new_max),
                      new_min, new_max)
              for b in bands]

    filled = [b.filled(fill_value=new_min) for b in scaled]
    if alpha:
        filled.append(_mask_to_alpha(scaled))

    return np.dstack(filled)




def draw_contours(img, contours, color=(0, 1, 0), thickness=2):
    n_img = img.copy()
    cv2.drawContours(n_img,contours,-1,color,thickness=thickness)
    return n_img


im_count = draw_contours(bands_to_display(get_rgb(load_bands(train_scene_filename)), alpha=False),
                         load_contours(train_ground_truth_filename, train_scene_filename))


plt.figure(1, figsize=(10,10))
plt.imshow(im_count)
_ = plt.title('Contours Drawn over Image')

##im_count





def create_contour_mask(contours, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (1), thickness=-1)
    return mask < 1

def combine_masks(masks):
    return np.any(np.dstack(masks), 2)

def add_mask(bands, mask):
    bands[0].mask = combine_masks([bands[0].mask, mask])

def mask_contours(bands, contours, in_contours=False):
    contour_mask = create_contour_mask(contours, bands[0].mask.shape)
    if in_contours:
        contour_mask = ~contour_mask
    _add_mask(bands, contour_mask)
    return bands




fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))

ax1.imshow(bands_to_display(get_rgb(mask_contours(load_bands(train_scene_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_scene_filename))),
                            alpha=True))
ax1.set_title('Crop Pixels')

img = bands_to_display(get_rgb(mask_contours(load_bands(train_scene_filename), load_contours(train_ground_truth_filename,train_scene_filename))),alpha=True)
#print(shape)
#ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# u_white = np.array([255,255,255])
# l_white = np.array([127,241,243])

im = rasterio.open(train_scene_filename)
mask = np.zeros(im.shape, dtype=np.uint8)
mask.fill(255)

roi = cv2.bitwise_and(img,img,mask = mask)

plt.imshow(roi)


def calc_area(im_rast, thresh):
    #for i in range(240,255):
    non_white_pixels = np.sum(thresh < 240)
    # non_white_pixels = cv2.findNonZero(thresh)

    white_pixels = np.sum(thresh >=240)
    #print(im.bounds[2])
    area_of_field = (non_white_pixels/(white_pixels + non_white_pixels))*(im.bounds[1]-im.bounds[3])*(im.bounds[2]-im.bounds[0])

    if area_of_field>0:
        print("Area is:" +str(area_of_field))
    else:
        print("Area is:" +str(-1*area_of_field))

calc_area(im, roi)


def classified_band_from_masks(band_mask, class_masks):
    class_band = np.zeros(band_mask.shape, dtype=np.uint8)

    for i, mask in enumerate(class_masks):
        class_band[~mask] = i

    return np.ma.array(class_band, mask=band_mask)


def create_contour_classified_band(pl_filename, ground_truth_filename):
    band_mask = load_bands(pl_filename)[0].mask
    contour_mask = create_contour_mask(load_contours(ground_truth_filename, pl_filename),
                                        band_mask.shape)
    return classified_band_from_masks(band_mask, [contour_mask, ~contour_mask])



#from scipy 2018, pla
def to_X(bands):
    return np.stack([b.compressed() for b in bands], axis=1)

def to_y(classified_band):
    return classified_band.compressed()





#neighobur = 3



#referenced from planet.com and researched
def fit_classifier(pl_filename, ground_truth_filename, metadata_filename):
    n_neighbors = 15
    weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(3, weights=weights)
    
    train_class_band = create_contour_classified_band(pl_filename, ground_truth_filename)
    X = to_X(load_refl_bands(pl_filename, metadata_filename))
    y = to_y(train_class_band)
    clf.fit(X, y)
    return clf



#metadata needed for testing
clf = fit_classifier(train_scene_filename,
                     train_ground_truth_filename,
                     train_metadata_filename
                     )




