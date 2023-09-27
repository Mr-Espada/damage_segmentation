import geojson
import shapely.wkt
import os
import numpy as np
import cv2
import sys

def jsonmaskpng(json_path, mask_path):
    '''
    Convert GeoJson to PNG mask for each class: minor-damage, major-damage, destroyed, no-damage
    mask_path: path where all the labels are (json files)
    output_path: path where to save all the pngs created
    '''
    os.makedirs(mask_path)
    path_jsons =  [ file for file in os.listdir(json_path) if file.endswith('.json') ]
    for json in path_jsons:
        path_to_file = os.path.join(json_path,json)
        with open(path_to_file) as file:
            gj = geojson.load(file)
        if "post" in json:
            mask_minor_damage = np.zeros((1024,1024)).astype(np.uint8)
            mask_major_damage = np.zeros((1024,1024)).astype(np.uint8)
            mask_destoryed = np.zeros((1024,1024)).astype(np.uint8)
        mask_no_damage = np.zeros((1024,1024)).astype(np.uint8)
        for i in range(len(gj['features']['xy'])):
            
            g1 = shapely.wkt.loads(gj['features']['xy'][i]['wkt'])
            g2 = geojson.Feature(geometry=g1, properties={})
            pts = np.array(g2.geometry["coordinates"]).astype(np.int32)
            if "pre" in json:
                mask_no_damage = cv2.fillPoly(mask_no_damage, pts*2, True, (255), 1)
                continue
            else:
                type = gj['features']['xy'][i]['properties']['subtype']
                if type == 'minor-damage':
                    mask_minor_damage = cv2.fillPoly(mask_minor_damage, pts*2, True, (255), 1)
                elif type == 'major-damage':
                    mask_major_damage = cv2.fillPoly(mask_major_damage, pts*2, True, (255), 1)
                elif type == 'destroyed':
                    mask_destoryed = cv2.fillPoly(mask_destoryed, pts*2, True, (255), 1)
                else:
                    mask_no_damage = cv2.fillPoly(mask_no_damage, pts*2, True, (255), 1)
             
        if "post" in json:            
            cv2.imwrite(os.path.join(mask_path, json.replace(".json", "_mask-minordamage.png")), mask_minor_damage*255)
            cv2.imwrite(os.path.join(mask_path, json.replace(".json", "_mask-majordamage.png")), mask_major_damage*255)
            cv2.imwrite(os.path.join(mask_path, json.replace(".json", "_mask-destoryed.png")), mask_destoryed*255)
        cv2.imwrite(os.path.join(mask_path, json.replace(".json", "_mask-nodamage.png")), mask_no_damage*255)


if __name__ == "__main__":

    '''
    mask path as first system argument (should contain train or test folder).
    output path as second system argument.
    args*: all the names of folders containing labels folder that holds all the jsons
    '''

    for path in sys.argv[3:]:
        json_path = os.path.join(path, "labels")
        mask_path = os.path.join(path, "masks")
        print(json_path)
        print(mask_path)
        jsonmaskpng(os.path.join(sys.argv[1], json_path), os.path.join(sys.argv[2], mask_path))


    