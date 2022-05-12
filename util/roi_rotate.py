from torch import nn
import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F

class ROIRotate(nn.Module):

    def __init__(self):

        super().__init__()
        self.height = 28
        self.width = 28
        self.restore_factor = 8
        
    def forward(self, feature, boxes, num_boxes):
        matrixes = []
        features = feature.repeat(num_boxes,1,1,1)
        boxes_width = []
        maximum_width = 0
        for box in boxes:
            x1, y1, x3, y3 = box[0].to('cpu').detach().numpy()
            x2 = x3
            y2 = y1
            x4 = x1
            y4 = y3
          
            rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2],[x3, y3], [x4, y4]]))
            box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

            if box_w <= box_h:
                box_w, box_h = box_h, box_w
            
            width = feature.size(3)
            height = feature.size(2)

            mapped_x1, mapped_y1 = (0, 0)
            mapped_x4, mapped_y4 = (0, self.height)

            width_box = math.ceil(self.height * box_w / box_h)
            width_box = min(width_box, width) 
            max_width = width_box if width_box > 0 else 0
            if max_width > maximum_width :
              maximum_width = max_width

            mapped_x2, mapped_y2 = (width_box, 0)

            src_pts = np.float32([(x1, y1), (x2, y2),(x4, y4)])
            dst_pts = np.float32([
              (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
            ])

            affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
            affine_matrix = ROIRotate.param2theta(affine_matrix, width, height)
            affine_matrix *= 1e10 
            affine_matrix = torch.tensor(affine_matrix, device=feature.device, dtype=torch.float)
            affine_matrix /= 1e10

            matrixes.append(affine_matrix)
            boxes_width.append(width_box)


        matrixes = torch.stack(matrixes)
        grid = F.affine_grid(matrixes, features.size(),align_corners = True)
        feature_rotated = F.grid_sample(features, grid,align_corners = True)

        channels = feature_rotated.shape[1]
        cropped_images_padded = torch.zeros((len(feature_rotated), channels, self.height, maximum_width),
                                            dtype=feature_rotated.dtype,device=feature_rotated.device)
        
        for i in range(feature_rotated.shape[0]):
            w = boxes_width[i]
            if maximum_width == w:
              cropped_images_padded[i] = feature_rotated[i, :, 0:self.height, 0:w]
            else:
              padded_part = torch.zeros((channels, self.height, maximum_width - w),
                                          dtype=feature_rotated.dtype,
                                          device=feature_rotated.device)
              cropped_images_padded[i] = torch.cat([feature_rotated[i, :, 0:self.height, 0: w], padded_part], dim=-1)

        lengths = np.array(boxes_width)
        indices = np.argsort(lengths) 
        indices = indices[::-1].copy() 
        lengths = lengths[indices]
        cropped_images_padded = cropped_images_padded[indices]
        recog_input =  F.interpolate(cropped_images_padded, size = (224,224),mode = 'bilinear',align_corners = True)
        
        return recog_input.to('cuda')

    @staticmethod
    def param2theta(param, w, h):
        param = np.vstack([param, [0, 0, 1]])
        try:
            param = np.linalg.inv(param)
        except Exception as e:
            print('fuck')
            raise e

        theta = np.zeros([2, 3])
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * h / w
        theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
        theta[1, 0] = param[1, 0] * w / h
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
        return theta


