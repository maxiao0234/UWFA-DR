import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageChops
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import AutoTokenizer
import json
from tqdm.auto import tqdm

from train_clip import CLIPTextModelPromptTuning, CLIPVisionModelPromptTuning
from train_multi_to_image_lora import FusionEmbeds


def main(args):
    device = 'cuda'
    prompt_source = "a ffa of a fundus"
    prompt_target = "a ffa of a normal fundus"
    prompt_uncond = "a ffa of a dr fundus"
    tokenizer = AutoTokenizer.from_pretrained(args.clip_model_id_org)
    text_encoder = CLIPTextModelPromptTuning.from_pretrained(args.clip_model_id_org).to(device)
    text_encoder_checkpoint = torch.load(os.path.join(args.output_dir, f'text_checkpoint_{args.fold}.pth'))
    text_encoder.load_state_dict(text_encoder_checkpoint)
    image_encoder = CLIPVisionModelPromptTuning.from_pretrained(args.clip_model_id_org).to(device)
    image_encoder_checkpoint = torch.load(os.path.join(args.output_dir, f'image_checkpoint_{args.fold}.pth'))
    image_encoder.load_state_dict(image_encoder_checkpoint)
    fusion_model = FusionEmbeds().to(device)
    fusion_model_checkpoint = torch.load(os.path.join(args.output_dir, f'fusion_checkpoint_{args.fold}.pth'))
    fusion_model.load_state_dict(fusion_model_checkpoint)

    scheduler = DDIMScheduler(beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                         scheduler=scheduler,
                                                         ).to(device)
    ldm_stable.scheduler.set_timesteps(args.num_steps)
    ldm_stable.load_lora_weights(os.path.join(args.output_dir, f'pytorch_lora_weights_{args.fold}.safetensors'))
    text_encoder.eval()
    image_encoder.eval()
    fusion_model.eval()

    uncond_input = tokenizer(
        [prompt_uncond],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_input_source = tokenizer(
        [prompt_source],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings_source = text_encoder(text_input_source.input_ids.to(device))[0]
    text_input_target = tokenizer(
        [prompt_target],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings_target = text_encoder(text_input_target.input_ids.to(device))[0]

    with torch.no_grad():
        with open(os.path.join(args.dataset_dir, f'metadata_test_{args.fold}.jsonl'), 'r') as f:
            for line in f:
                d = json.loads(line)
                file_name = os.path.join(args.dataset_dir, d['file_name'])
                image_org = np.array(Image.open(file_name).convert('RGB'))
                embeds = torch.load(os.path.join(args.dataset_dir, 'image_embeds', f'fold{args.fold}_' + d['embeds_name'])).unsqueeze(0).to(device)

                text_embeddings_source_fusion = fusion_model(text_embeddings_source, embeds)
                text_embeddings_target_fusion = fusion_model(text_embeddings_target, embeds)
                uncond_embeddings_fusion = fusion_model(uncond_embeddings, embeds)

                image = torch.from_numpy(image_org).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to('cuda')
                latent = ldm_stable.vae.encode(image)['latent_dist'].mean
                latent = latent * 0.18215
                last_latent = latent.clone().detach()

                # forward
                for i in tqdm(range(args.num_steps)):
                    t = ldm_stable.scheduler.timesteps[len(ldm_stable.scheduler.timesteps) - i - 1]

                    noise_pred = ldm_stable.unet(last_latent, t,
                                                 encoder_hidden_states=text_embeddings_source_fusion)["sample"]

                    t, next_timestep = min(
                        t - ldm_stable.scheduler.config.num_train_timesteps // ldm_stable.scheduler.num_inference_steps,
                        999), t
                    alpha_prod_t = ldm_stable.scheduler.alphas_cumprod[
                        t] if t >= 0 else ldm_stable.scheduler.final_alpha_cumprod
                    alpha_prod_t_next = ldm_stable.scheduler.alphas_cumprod[next_timestep]
                    beta_prod_t = 1 - alpha_prod_t
                    next_original_sample = (last_latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
                    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
                    last_latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

                # inversion
                for i, t in enumerate(tqdm(ldm_stable.scheduler.timesteps[-args.num_steps:])):
                    noise_pred_uncond = ldm_stable.unet(last_latent, t,
                                                 encoder_hidden_states=text_embeddings_source)["sample"]
                    noise_prediction = ldm_stable.unet(last_latent, t,
                                                        encoder_hidden_states=text_embeddings_target)["sample"]
                    noise_pred = noise_pred_uncond + 7.5 * (noise_prediction - noise_pred_uncond)
                    last_latent = ldm_stable.scheduler.step(noise_pred, t, last_latent)["prev_sample"]

                last_latent = 1 / 0.18215 * last_latent
                images_inv = ldm_stable.vae.decode(last_latent)['sample']
                images_inv = (images_inv / 2 + 0.5).clamp(0, 1)
                images_inv = images_inv.cpu().permute(0, 2, 3, 1).numpy()
                images_inv = (images_inv * 255).astype(np.uint8)

                image_org = Image.fromarray(image_org).convert('L')
                images_inv = Image.fromarray(images_inv[0]).convert('L')
                image_diff = ImageChops.difference(image_org, images_inv)
                save_image = Image.new('L', (512 * 3, 512), 'white')
                save_image.paste(image_org, (0, 0))
                save_image.paste(images_inv, (512, 0))
                save_image.paste(image_diff, (512 * 2, 0))
                save_name = d['file_name'].split('/')[1] + '_' + d['file_name'].split('/')[2]
                print(os.path.join(args.result_dir, save_name))
                save_image.save(os.path.join(args.result_dir, save_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UWFA Inversion script', add_help=False)
    parser.add_argument('--project-name', default='UWFA Inversion', type=str)
    parser.add_argument('--dataset-dir', default='', type=str)
    parser.add_argument('--output-dir', default='', type=str)
    parser.add_argument('--result-dir', default='', type=str)
    parser.add_argument('--pretrained_model_name_or_path', default='', type=str)
    parser.add_argument('--clip_model_id_org', default='', type=str)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--num_steps', default=50, type=int)
    args = parser.parse_args()

    main(args)
