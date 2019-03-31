# #rice
# import cv2
# import numpy as np

# fixed_size = tuple((500, 500))

# im = cv2.imread("../input/set123/rice.jpg")
# hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# l_green = np.array([65,60,60])
# u_green = np.array([80,255,255])
# mask = cv2.inRange(hsv, l_green, u_green)

# import matplotlib.pyplot as plt

# #plt.imshow(mask)

# res = cv2.bitwise_and(hsv,hsv, mask=mask)
# # #import matplotlib.pyplot as plt

# plt.imshow(res)

# _,threshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
# #plt.imshow(threshed,cmap='gray')
# threshed = cv2.dilate(threshed)

# closing = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

# # contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# plt.imshow(closing,cmap='gray')

# #training
# #1->resize
# #2->normalize

# #im_path=""
# ima = cv2.bitwise_and(im,threshed,mask = mask )
# plt.imshow(ima)

# contours,hierarchy = cv2.findContours(ima, 1, 2)
# plt.imshow(contours)

# area = cv2.contourArea(contours)
# print(area)
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report

# from sklearn.model_selection import train_test_split


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

print(len(load_contours(train_ground_truth_filename, train_scene_filename)))



#Referenced
N_bs = namedtuple('N_bs', 'b, g, r, nir')

#Not this one
def load_col_bands(filename):
    with rasterio.open(filename) as src:
        b, g, r, nir = src.read()
        mask = src.read_masks(1) == 0 # 0 value means the pixel is masked
    
    col_bands = N_bs(b=b, g=g, r=r, nir=nir)
    return N_bs(*[np.ma.array(b, mask=mask)
                        for b in col_bands])

def r_g_b(band):
    return [band.r, band.g, band.b]

def check_mask(col_band):
    return '{}/{} masked'.format(col_band.mask.sum(), col_band.mask.size)
    
print(check_mask(load_col_bands(train_scene_filename).r))



NamedCoefficients = namedtuple('NamedCoefficients', 'b, g, r, nir')

#scipy 2018
def read_refl_coefficients(metadata_filename):
    xmldoc = minidom.parse(metadata_filename)
    nodes = xmldoc.getElementsByTagName("ps:col_bandSpecificMetadata")

    
    coeffs = {}
    for node in nodes:
        n = node.getElementsByTagName("ps:col_bandNumber")[0].firstChild.data
        if n in ['1', '2', '3', '4']:
            i = int(n)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    return NamedCoefficients(b=coeffs[1],
                             g=coeffs[2],
                             r=coeffs[3],
                             nir=coeffs[4])


#Ref from planet/explorer
def load_refl_col_bands(filename, metadata_filename):
    col_bands = load_col_bands(filename)
    coeffs = read_refl_coefficients(metadata_filename)
    return N_bs(*[b.astype(float)*c for b,c in zip(col_bands, coeffs)])


print(read_refl_coefficients(train_metadata_filename))



#NDVI
def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
    return (ndarray - old_min)*(new_max - new_min)/(old_max - old_min) + new_min

def gen_mask(col_bands):
    col_band = np.atleast_3d(col_bands[0])
    alpha = np.zeros_like(col_band)
    alpha[~col_band.mask] = 1
    return alpha

def _add_alpha_mask(col_bands):
    return np.dstack([col_bands, gen_mask(col_bands)])

#Referenced from planet.com
def col_bands_to_display(col_bands, alpha=False):
#     """Converts a list of 4 col_bands to a 3-col_band rgb, normalized array for display."""  
    assert len(col_bands) in [1,3]
    all_col_bands = np.dstack(col_bands)
    old_min = np.percentile(all_col_bands, 2)
    old_max = np.percentile(all_col_bands, 98)

    new_min = 0
    new_max = 1
    scaled = [np.clip(_linear_scale(b.astype(np.double),
                                    old_min, old_max, new_min, new_max),
                      new_min, new_max)
              for b in col_bands]

    filled = [b.filled(fill_value=new_min) for b in scaled]
    if alpha:
        filled.append(gen_mask(scaled))

    return np.dstack(filled)




def draw_contours(img, contours, color=(0, 1, 0), thickness=2):
    n_img = img
    cv2.drawContours(n_img,contours,-1,color,thickness=thickness)
    return n_img


im_count = draw_contours(col_bands_to_display(r_g_b(load_col_bands(train_scene_filename)), alpha=False),
                         load_contours(train_ground_truth_filename, train_scene_filename))


plt.figure(1, figsize=(10,10))
plt.imshow(im_count)
_ = plt.title('Contours Drawn over Image')

##im_count





def create_contour_mask(contours, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (1), thickness=-1)
    return mask < 1
#     ^--=====

def combine_masks(masks):
    return np.any(np.dstack(masks), 2)

def add_mask(col_bands, mask):
    col_bands[0].mask = combine_masks([col_bands[0].mask, mask])

def mask_contours(col_bands, contours, in_contours=False):
    contour_mask = create_contour_mask(contours, col_bands[0].mask.shape)
    if in_contours:
        contour_mask = ~contour_mask
    _add_mask(col_bands, contour_mask)
    return col_bands




fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))

ax1.imshow(col_bands_to_display(r_g_b(mask_contours(load_col_bands(train_scene_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_scene_filename))),
                            alpha=True))
ax1.set_title('Crop Pixels')

ax2.imshow(col_bands_to_display(r_g_b(mask_contours(load_col_bands(train_scene_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_scene_filename),
                                                  in_contours=True)),
                            alpha=True))
ax2.set_title('Non-Crop Pixels')
plt.tight_layout()





def classified_col_band_from_masks(col_band_mask, class_masks):
    class_col_band = np.zeros(col_band_mask.shape, dtype=np.uint8)

    for i, mask in enumerate(class_masks):
        class_col_band[~mask] = i

    # turn into masked array, using col_band_mask as mask
    return np.ma.array(class_col_band, mask=col_band_mask)


def create_contour_classified_col_band(pl_filename, ground_truth_filename):
    col_band_mask = load_col_bands(pl_filename)[0].mask
    contour_mask = create_contour_mask(load_contours(ground_truth_filename, pl_filename),
                                        col_band_mask.shape)
    return classified_col_band_from_masks(col_band_mask, [contour_mask, ~contour_mask])

# plt.figure(1, figsize=(10,10))
# plt.imshow(create_contour_classified_col_band(train_scene_filename, train_ground_truth_filename))
# _ = plt.title('Ground Truth Classes')



#from scipy 2018, pla
def to_X(col_bands):
    return np.stack([b.compressed() for b in col_bands], axis=1)

def to_y(classified_col_band):
    return classified_col_band.compressed()




def save_cross_val_data(pl_filename, ground_truth_filename, metadata_filename):
    train_class_col_band = create_contour_classified_col_band(pl_filename, ground_truth_filename)
    X = to_X(load_refl_col_bands(pl_filename, metadata_filename))
    y = to_y(train_class_col_band)

    cross_val_dir = os.path.join('data', 'knn_cross_val')
    if not os.path.exists(cross_val_dir):
        os.makedirs(cross_val_dir)

    xy_file = os.path.join(cross_val_dir, 'xy_file.npz')
    np.savez(xy_file, X=X, y=y)
    return xy_file

print(save_cross_val_data(train_scene_filename,
                          train_ground_truth_filename,
                          train_metadata_filename))


#neighobur = 3






