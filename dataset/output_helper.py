import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
#basic info about limb composition
joint_to_limb_heatmap_relationship = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17]]
#for plot usage
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]



def draw_pose_figure(coors,height=360,width=640,limb_thickness=4):
    canvas = np.ones([height,width,3])*255
    canvas = canvas.astype(np.uint8)
    limb_type = 0
    for joint_relation in  joint_to_limb_heatmap_relationship:
        if(limb_type >= 17):
            break
        joint_coords = coors[joint_relation]
        for joint in joint_coords:  # Draw circles at every joint
            '''
            haoran added print
            '''
            #print('joint',joint)
            cv2.circle(canvas, tuple(joint[0:2].astype(
                        int)), 4, (0,0,0), thickness=-1)  
        coords_center = tuple(
                    np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0, :] - joint_coords[1, :]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(
                    coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[limb_type])
        limb_type += 1
    return canvas


def imshow(image):
    plt.imshow(image[:,:,[2,1,0]].astype(np.uint8))
    plt.show()
    
def draw_pose_figure_for_grid(coors,height=int(360/8),width=int(640/8)):
    limb_thickness = 2
    canvas = np.ones([height,width,3])*255
    canvas = canvas.astype(np.uint8)
    limb_type = 0
    coors = coors.astype(np.uint8)
    for joint_relation in  joint_to_limb_heatmap_relationship:
        if(limb_type >= 17):
            break
        joint_coords = coors[joint_relation]
        for joint in joint_coords:  # Draw circles at every joint
            '''
            haoran added print
            '''
            #print('joint',joint)
            cv2.circle(canvas, tuple(joint[0:2].astype(
                        int)), 2, (0,0,0), thickness=-1)  
        coords_center = tuple(
                    np.round(np.mean(joint_coords, 0)).astype(int))
        limb_dir = joint_coords[0, :] - joint_coords[1, :]
        limb_length = np.linalg.norm(limb_dir)
        angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
        polygon = cv2.ellipse2Poly(
                    coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[limb_type])
        limb_type += 1
    imshow(canvas)
    return canvas
    
'''
coors should be 50*18*2 size array
'''
import os
def makemydir(whatever):
    try:
        os.makedirs(whatever)
    except OSError:
        pass
  # let exception propagate if we just can't
  # cd into the specified directory
    os.chdir(whatever)
    
def save_batch_images(coors,batch_num,save_dir_start='/mnt/external4/output_demo'):
    reshape_coors = coors.reshape([50,18,2])
    for i in range(reshape_coors.shape[0]):
        idx = str("%03d" % i)
        
        target_dir = save_dir_start + '/' + str(batch_num)
        
        makemydir(target_dir)
        output_dir = target_dir +'/'+idx+'.jpeg'
        img = draw_pose_figure(reshape_coors[i])
        cv2.imwrite(output_dir,img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    
def save_batch_images_long(coors,batch_num,save_dir_start='/mnt/external4/output_demo'):
    reshape_coors = coors.reshape([300,18,2])
    for i in range(reshape_coors.shape[0]):
        idx = str("%03d" % i)
        
        target_dir = save_dir_start + '/' + str(batch_num)
        
        makemydir(target_dir)
        output_dir = target_dir +'/'+idx+'.jpeg'
        img = draw_pose_figure(reshape_coors[i])
        cv2.imwrite(output_dir,img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    
    
def save_2_batch_images(real_coors,fake_coors,batch_num,save_dir_start='/mnt/external4/output_demo'):
    real_reshape_coors = real_coors.reshape([50,18,2])
    fake_reshape_coors = fake_coors.reshape([50,18,2])
    for i in range(real_reshape_coors.shape[0]):
        idx = str("%03d" % i)
        
        target_dir = save_dir_start + '/' + str(batch_num)
        
        makemydir(target_dir)
        output_dir = target_dir +'/'+idx+'.jpeg'
        real_img = draw_pose_figure(real_reshape_coors[i])
        fake_img = draw_pose_figure(fake_reshape_coors[i])
        img = np.vstack([real_img,fake_img])
        
        cv2.imwrite(output_dir,img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
        
def save_2_batch_images_long(real_coors,fake_coors,batch_num,save_dir_start='/mnt/external4/output_demo'):
    real_reshape_coors = real_coors.reshape([300,18,2])
    fake_reshape_coors = fake_coors.reshape([300,18,2])
    for i in range(real_reshape_coors.shape[0]):
        idx = str("%03d" % i)
        
        target_dir = save_dir_start + '/' + str(batch_num)
        
        makemydir(target_dir)
        output_dir = target_dir +'/'+idx+'.jpeg'
        real_img = draw_pose_figure(real_reshape_coors[i])
        fake_img = draw_pose_figure(fake_reshape_coors[i])
        img = np.vstack([real_img,fake_img])
        
        cv2.imwrite(output_dir,img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
        
def demo_save_batch_images(coors,rows=50,cols=1):
    reshape_coors = coors.reshape([rows,cols,18,2])
    reshape_coors = np.array(reshape_coors)
    for row in range(rows):
        
        
        for col in range(cols):
            if(col == 0):
                col_img = draw_pose_figure(reshape_coors[row,col])
                
                col_img = cv2.resize(col_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                imshow(col_img)
            else:
                col_img = np.hstack([col_img,reshape_coors[row,col]])
                
        if(row == 0):
            row_img = col_img
        else:
            row_img = np.vstack([row_img,col_img])
            
    imshow(row_img)
              