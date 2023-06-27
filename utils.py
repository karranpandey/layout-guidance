import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os

def save_attn_img(attn_map, name):
    upscale_ratio = 512 / attn_map.shape[1]
    attn_map = attn_map.detach().cpu().numpy() 
    attn_map = attn_map.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = (attn_map * 255).round().astype("uint8")
    img = Image.fromarray(attn_map)
    img.save('./example_output/attn_map_' + name + '.png')
    return

def normalize_attn_torch(attn_map):
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = 10*(attn_map - 0.5)
    attn_map = torch.sigmoid(attn_map)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map

def normalize_attn(attn_map):
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = 10*(attn_map - 0.5)
    attn_map = 1 / (1 + np.exp(-attn_map))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map
    
def compute_appearance_loss(attn_maps_mid, attn_maps_up, activations, filtered_act_orig, obj_idx, object_positions):
    #normalize attention maps 
    attn_map = 0
    for attn_map_integrated in attn_maps_up[0]:
        attn_map += attn_map_integrated
        
    attn_map /= len(attn_maps_up[0])
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    ca_map_obj = 0

    for object_position in object_positions[obj_idx]:
        ca_map_obj += attn_map[:,:,object_position].reshape(b,H,W)

    ca_map_obj = ca_map_obj.mean(axis = 0)
    ca_map_obj = normalize_attn_torch(ca_map_obj)
    ca_map_obj = ca_map_obj.view(1, 1, H, W)
    m = nn.Upsample(scale_factor=activations.shape[2] / H, mode='nearest')
    ca_map_obj = m(ca_map_obj)
    
    #find filtered activations 
    filtered_act = torch.mul(ca_map_obj, activations)

    #find L-1 norm
    loss = torch.mean(filtered_act - filtered_act_orig)
    return loss

def compute_ca_loss(attn_maps_mid, attn_maps_up, bboxes, object_positions):
    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated.chunk(2)[1]

        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:

                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

                #for i in range(b):
                #    save_attn_img(ca_map_obj[i], str(i))
                
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss/len(object_positions[obj_idx]))

    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated.chunk(2)[1]
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                # ca_map_obj = attn_map[:, :, object_positions[obj_position]].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss

def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions

def draw_box(pil_img, bboxes, phrases, save_path):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('./FreeMono.ttf', 25)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
            draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=font, fill=(255, 0, 0))
    pil_img.save(save_path)



def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
