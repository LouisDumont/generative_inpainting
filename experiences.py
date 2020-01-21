# --- Library Imports ---#
import cv2
import numpy as np
import os

# --- Global variables --- #
data_folder = os.getcwd() + '/examples/'
width, height = 680, 512

# --- Experiment Variables --- #
source_folder = '/own_raw/'
destination_folder = '/own/'
hole_sizes = [70, 110, 150]
hole_locations = [(40,100)] # keep len to 1, other lenghts are not supported
quadruple = True # Whether to multiply the experience by replicating the hole on all possible invertions of the positions
mirror = True # Whether to include mirror images


# --- Utils functions --- #
def compute_nb_repet(quadruple, mirror, hole_locations, hole_sizes):
    '''
    From the experiment parameters, computes the number of times each image is used in the experiments
    '''
    mirror_fac = mirror + 1 # equals 2 when mirror is True, else 1 
    quadruple_fac = 3 * quadruple +1 # equals 4 if quadruple is True, else 1
    return len(hole_sizes) * len(hole_locations) * mirror_fac * quadruple_fac


def copy_folder(source, dest, width, height, nb_repet):
    '''
    Resizes and renames the images from the source folder into a new folder
    '''
    filenames_source = os.listdir(source)

    if not os.path.isdir(dest):
        os.mkdir(dest)

    for i, name in enumerate(filenames_source):
        img_own = cv2.imread(source+filenames_source[i])
        img_resized = cv2.resize(img_own, (width,height))
        print('writting inder:', i*nb_repet)
        cv2.imwrite(dest + '/ex'+str(i*nb_repet)+'_raw.png', img_resized)

def create_mirrors(img):
    '''
    Creates an image where the right side of the original image is replicated twice
    '''
    height, width = img.shape[:2]
    mirror_img = img.copy()
    mirror_img[:,:width//2] = mirror_img[:, width//2:]
    return mirror_img

def create_inputs_and_masks(in_folder, dest_folder, quadruple, mirror, hole_sizes, hole_locations, nb_repet):
    '''
    From each original image, creates the input images and masks corresponding to experiment parameters
    '''
    images = os.listdir(dest_folder)
    nb_images = len(os.listdir(in_folder))

    assert nb_images > 0

    height, width = cv2.imread(dest_folder + '/' + images[0]).shape[:2]

    nb_masks = nb_repet
    if mirror: nb_masks = nb_masks//2
    mask_template = mask_templates = [np.zeros((height,width,3), np.uint8) for i in range(nb_masks)]

    nb_sizes = len(hole_sizes)
    pos_h, pos_w = hole_locations[0]
    for k, hole_size in enumerate(hole_sizes):            
            if quadruple:
                mask_templates[4*k][pos_h:pos_h+hole_size, pos_w:pos_w+hole_size] = 255
                mask_templates[4*k+1][height-pos_h-hole_size:height-pos_h, pos_w:pos_w+hole_size] = 255
                mask_templates[4*k+2][pos_h:pos_h+hole_size, width-pos_w-hole_size:width-pos_w] = 255
                mask_templates[4*k+3][height-pos_h-hole_size:height-pos_h, width-pos_w-hole_size:width-pos_w] = 255
            else:
                mask_templates[k][pos_h:pos_h+hole_size, pos_w:pos_w+hole_size] = 255

    count = 0
    for i in range(nb_images):
        print('reading inder:', i*nb_repet)
        img = cv2.imread(dest_folder + '/ex'+str(i*nb_repet)+'_raw.png')

        for mask_template in mask_templates:

            img0 = img.copy()
            img0[np.where(mask_template==255)] = 255
            print('writting inder:', count)
            cv2.imwrite(dest_folder + '/ex'+str(count)+'_input.png', img0)
            cv2.imwrite(dest_folder + '/ex'+str(count)+'_mask.png', mask_template)
            count += 1

            if mirror:
                img0 = create_mirrors(img.copy())
                img0[np.where(mask_template==255)] = 255
                print('writting inder:', count)
                cv2.imwrite(dest_folder + '/ex'+str(count)+'_input.png', img0)
                cv2.imwrite(dest_folder + '/ex'+str(count)+'_mask.png', mask_template)
                count += 1

# --- Main execution --- #
if __name__ == '__main__':
    source = data_folder + source_folder
    dest = data_folder + destination_folder

    nb_repet = compute_nb_repet(quadruple, mirror, hole_locations, hole_sizes)
    print('nb_repet', nb_repet)

    copy_folder(source, dest, width, height, nb_repet)

    create_inputs_and_masks(source, dest, quadruple, mirror, hole_sizes, hole_locations, nb_repet)

    nb_images = len(os.listdir(source))

    for i in range(nb_images*nb_repet):
        print('Processing image '+str(i))
        command = 'python test.py --image '+dest+'/ex'+str(i)+'_input.png --mask '+dest+'/ex'+str(i)+'_mask.png --output '+dest+'/ex'+str(i)+'_output.png --checkpoint_dir model_logs/release_places2_256'
        os.system(command)