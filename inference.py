import torch
import numpy as np
import math
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import compute_ca_loss, Pharse2idx, draw_box, setup_logger, compute_appearance_loss
import hydra
import os
from tqdm import tqdm

def save_act_img(act, name):
    upscale_ratio = 512 / act.shape[1]
    act_map = act.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
    act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())
    act_map = (act_map * 255).round().astype("uint8")
    img = Image.fromarray(act_map)
    img.save('./example_output/' + name + '_map.png')
    return

def save_attn_img(attn_map, obj_idx, object_positions):
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    return_attn_map = 0
    for obj_position in object_positions[obj_idx]:
        ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
        cum_attn_map = ca_map_obj.mean(axis = 0)
        cum_attn_map = normalize_attn(cum_attn_map)
        upscale_ratio = 512 / cum_attn_map.shape[1]
        cum_attn_map = cum_attn_map.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
        return_attn_map = cum_attn_map
        cum_attn_map = (cum_attn_map * 255).round().astype("uint8")
        img = Image.fromarray(cum_attn_map)
        img.save('./example_output/obj' + str(obj_idx) + '_attn_map.png')
        #for i in range(b):
        #    save_act_img(ca_map_obj[i].detach().cpu().numpy(), 'attn_obj' + str(obj_idx) + '_' + str(i))
    return return_attn_map

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

def compute_filtered_act(attn_maps_up, activations, obj_idx, object_positions):
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
    m = torch.nn.Upsample(scale_factor=activations.shape[2] / H, mode='nearest')
    ca_map_obj = m(ca_map_obj)
    
    #find filtered activations 
    filtered_act = torch.mul(ca_map_obj, activations)
    return filtered_act   
    
def filtered_act(attn_map, activations):
    upscale_ratio = 512 / activations.shape[1]
    act_map = activations.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
    act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())
    filtered_act = np.multiply(attn_map, act_map)
    upscale_ratio = 512 / filtered_act.shape[1]
    cum_attn_map = filtered_act.repeat(upscale_ratio, axis = 0).repeat(upscale_ratio, axis = 1)
    cum_attn_map = (cum_attn_map * 255).round().astype("uint8")
    img = Image.fromarray(cum_attn_map)
    img.save('./example_output/_filtered_act_map.png')


def retrieve_info(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, cfg, logger, examples):

    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")

    # Get Object Positions

    logger.info("Conver Phrases to Object Positions")
    object_positions = Pharse2idx(prompt, phrases)


    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the inital latent noise

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)


    latents = latents * noise_scheduler.init_noise_sigma


    loss = torch.tensor(10000)


    filtered_acts = []


    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        with torch.no_grad():
            latent_model_input = torch.cat([latents]*2)


            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down, activations = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)


            filtered_act = compute_filtered_act(attn_map_integrated_up, activations[0], 0, object_positions)


            filtered_acts.append(filtered_act)


            noise_pred = noise_pred.sample


            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)


            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()


    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for index, pil_image in enumerate(pil_images):
            image_path = os.path.join(cfg.general.save_path, 'example_orig_{}.png'.format(index))
            logger.info('save example image to {}'.format(image_path))
            draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)
       
    return filtered_acts


def inference(device, unet, vae, tokenizer, text_encoder, prompt, bboxes, phrases, filtered_acts, cfg, logger):


    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")

    # Get Object Positions

    logger.info("Conver Phrases to Object Positions")
    object_positions = Pharse2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the inital latent noise

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss = torch.tensor(10000)
    timestep_num = 0

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0
        filtered_act_orig = filtered_acts[timestep_num]
        while loss.item() / cfg.inference.loss_scale > 0.00001 and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = torch.cat([latents]*2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down, activations = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            # update latents with guidance
            #loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   #object_positions=object_positions) * cfg.inference.loss_scale

            loss = compute_appearance_loss(attn_map_integrated_mid, attn_map_integrated_up, activations[0], filtered_act_orig, 0, object_positions) * cfg.inference.loss_scale

            print(loss)
            
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2
            iteration += 1
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents]*2) #torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down, activations = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):

    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)



    # ------------------ example input ------------------
    examples = {"prompt": "A hello kitty toy is playing with a purple ball.",
                "phrases": "hello kitty; ball",
                "bboxes": [[[0.1, 0.2, 0.5, 0.8]], [[0.75, 0.6, 0.95, 0.8]]],
                'save_path': cfg.general.save_path
                }
    # ---------------------------------------------------
    # Prepare the save path
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)

    logger.info(cfg)
    # Save cfg
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

    # Inference
    filtered_acts = retrieve_info(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'], examples['phrases'], cfg, logger, examples)
    pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['bboxes'], examples['phrases'], filtered_acts, cfg, logger)

    # Save example images
    for index, pil_image in enumerate(pil_images):
        image_path = os.path.join(cfg.general.save_path, 'example_{}.png'.format(index))
        logger.info('save example image to {}'.format(image_path))
        draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)

if __name__ == "__main__":
    main()
