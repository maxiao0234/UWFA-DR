import os
import argparse
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from transformers import CLIPVisionModel, CLIPTextModel, CLIPImageProcessor, CLIPModel, AutoTokenizer
from transformers.models.clip.modeling_clip import CLIPVisionConfig, CLIPTextConfig, CLIP_VISION_INPUTS_DOCSTRING, CLIP_TEXT_INPUTS_DOCSTRING
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPooling
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, accuracy
from tqdm.auto import tqdm
# from tqdm import tqdm

from read_data import UWFADataset
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP Tuning script', add_help=False)
    parser.add_argument('--project-name', default='CLIP Tuning', type=str)
    parser.add_argument('--output-dir', default='', type=str)
    parser.add_argument('--dataset-dir', default='', type=str)
    parser.add_argument('--model_id_org', default='', type=str)

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    return parser


# def run_one_step(image_encoder, data_early, data_late):
#     out_early = image_encoder(data_early)
#     last_hidden_state_early = out_early.last_hidden_state
#     grade_embeds_early = last_hidden_state_early[:, 1, :]
#     period_embeds_early = last_hidden_state_early[:, 2, :]
#     side_embeds_early = last_hidden_state_early[:, 3, :]
#     hidden_states_early = out_early.hidden_states
#
#     out_late = image_encoder(data_late)
#     last_hidden_state_late = out_late.last_hidden_state
#     grade_embeds_late = last_hidden_state_late[:, 1, :]
#     period_embeds_late = last_hidden_state_late[:, 2, :]
#     side_embeds_late = last_hidden_state_late[:, 3, :]
#     hidden_states_late = out_late.hidden_states
#
#     state_grade = image_encoder.grade_proj(torch.cat([grade_embeds_early, grade_embeds_late], dim=-1))
#     prediction_grade = image_encoder.grade_head(state_grade)
#     state_period = image_encoder.period_proj(torch.cat([period_embeds_early, period_embeds_late], dim=0))
#     prediction_period = image_encoder.period_head(state_period)
#     state_side = image_encoder.side_proj(torch.cat([side_embeds_early, side_embeds_late], dim=0))
#     prediction_side = image_encoder.side_head(state_side)
#
#     return hidden_states_early, hidden_states_late, prediction_grade, prediction_period, prediction_side


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class CLIPVisionModelPromptTuning(CLIPVisionModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig, hidden_length=64):
        super().__init__(config)
        self.vision_model.requires_grad_(False)

        self.hidden_length = hidden_length
        for i in range(self.config.num_hidden_layers):
            setattr(self, f'prompt_embedding_{i}', nn.Parameter(torch.randn(1, hidden_length, self.config.hidden_size)))

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch_size = pixel_values.shape[0]
        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        encoder_states = (hidden_states,)

        for idx, encoder_layer in enumerate(self.vision_model.encoder.layers):
            prompt_embeds = getattr(self, f'prompt_embedding_{idx}').expand(batch_size, -1, -1)
            # cls_token = hidden_states[:, 0: 1, :]
            # if idx == 0:
            #     hidden_states = torch.cat([cls_token, prompt_embeds, hidden_states[:, 1:, :]], dim=1)
            # else:
            #     hidden_states = torch.cat([cls_token, prompt_embeds, hidden_states[:, self.hidden_length + 1:, :]], dim=1)

            hidden_states = torch.cat([prompt_embeds, hidden_states], dim=1)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0][:, self.hidden_length:, :]
            encoder_states = encoder_states + (hidden_states,)

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        # if not return_dict:
        #     return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_states,
            # attentions=encoder_outputs.attentions,
        )


class CLIPTextModelPromptTuning(CLIPTextModel):
    config_class = CLIPTextConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig, hidden_length=16):
        super().__init__(config)
        self.text_model.requires_grad_(False)

        self.hidden_length = hidden_length
        for i in range(self.config.num_hidden_layers):
            setattr(self, f'prompt_embedding_{i}', nn.Parameter(torch.randn(1, hidden_length, self.config.hidden_size)))

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        # encoder_outputs = self.encoder(
        #     inputs_embeds=hidden_states,
        #     attention_mask=attention_mask,
        #     causal_attention_mask=causal_attention_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        encoder_states = (hidden_states, )

        for idx, encoder_layer in enumerate(self.text_model.encoder.layers):
            prompt_embeds = getattr(self, f'prompt_embedding_{idx}').expand(batch_size, -1, -1)
            hidden_states = torch.cat([prompt_embeds, hidden_states], dim=1)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0][:, self.hidden_length:, :]
            encoder_states = encoder_states + (hidden_states,)

        last_hidden_state = hidden_states
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        if self.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        # if not return_dict:
        #     return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_states,
            # attentions=encoder_outputs.attentions,
        )


# def infoNCE_loss(embedding_1, embedding_2, temperature=0.5):
#     embedding_1 = embedding_1.reshape(-1, embedding_1.shape[-1])
#     embedding_2 = embedding_2.reshape(-1, embedding_2.shape[-1])
#     similarity_matrix = torch.matmul(embedding_1, embedding_2.t()) / temperature
#     labels = torch.arange(embedding_1.shape[0]).to(embedding_1.device)
#     positives = similarity_matrix[labels, labels].unsqueeze(1)
#     negatives = similarity_matrix.clone()
#     negatives.fill_diagonal_(float('-inf'))
#     negatives = torch.logsumexp(negatives, dim=1, keepdim=True)
#
#     logits_diff = positives - negatives
#     return -logits_diff.mean()


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(text_embeds, image_embeds, logit_scale):
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
    caption_loss = contrastive_loss(logits_per_text)
    image_loss = contrastive_loss(logits_per_text.t())
    return (caption_loss + image_loss) / 2.0


# def clip_loss_multi(text_embeds, image_embeds_list, logit_scale):
#     N = len(image_embeds_list)
#     B, C = image_embeds_list[0].shape
#     image_embeds = []
#     for embeds in image_embeds_list:
#         for n in range(N - 1):
#             embeds = torch.stack([embeds] * B, dim=0)
#         image_embeds.append(embeds)
#     image_embeds = torch.stack(image_embeds, dim=-2)
#     image_embeds = image_embeds.reshape((B ** N, N, C))
#
#     text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
#     image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
#
#     logits_per_text = torch.einsum('g c, b h c -> b g h', text_embeds, image_embeds) * logit_scale
#     logits_per_image = logits_per_text.permute(0, 2, 1)
#     logits_per_text = logits_per_text.reshape((B ** N * N, N))
#     logits_per_image = logits_per_image.reshape((B ** N * N, N))
#
#     label = torch.arange(N, device=logits_per_text.device).unsqueeze(0).expand(B ** N, -1).reshape(-1)
#     caption_loss = nn.functional.cross_entropy(logits_per_text, label)
#     image_loss = nn.functional.cross_entropy(logits_per_image, label)
#
#     loss = (caption_loss + image_loss) / 2.0
#     acc = accuracy(logits_per_image, label)[0]
#
#     return loss, acc


def run_image_embeds(pixel_values, image_encoder, visual_projection, hidden_length=64):
    image_outs = image_encoder(pixel_values)
    image_embeds = image_outs[0]
    image_embeds = image_embeds[:, 1:, :]
    image_embeds = image_embeds.reshape(image_embeds.shape[0], -1)
    image_embeds_pooling = image_outs[1]
    image_embeds_pooling = visual_projection(image_embeds_pooling)
    return image_embeds, image_embeds_pooling


def run_text_embeds(prompts, tokenizer, text_model, text_projection, device='cuda'):
    text_embeds = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
    text_embeds = text_model(**text_embeds)
    text_embeds = text_embeds[1]
    text_embeds = text_projection(text_embeds)
    return text_embeds


def train(args):
    dataset_train = UWFADataset(
        args.dataset_dir,
        args.model_id_org,
        fold=args.fold,
        split='train',
        is_train=True
    )
    sampler_train = RandomSampler(dataset_train)
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    dataset_test = UWFADataset(
        args.dataset_dir,
        args.model_id_org,
        fold=args.fold,
        split='test',
        is_train=False
    )
    clip_model = CLIPModel.from_pretrained(args.model_id_org).requires_grad_(False).to('cuda')
    visual_projection = clip_model.visual_projection
    text_projection = clip_model.text_projection
    logit_scale = clip_model.logit_scale.exp()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_org)

    class TuningModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = CLIPTextModelPromptTuning.from_pretrained(args.model_id_org)
            self.image_encoder = CLIPVisionModelPromptTuning.from_pretrained(args.model_id_org)
    encoders = TuningModel()
    text_encoder = encoders.text_encoder
    image_encoder = encoders.image_encoder

    n_parameters = sum(p.numel() for p in encoders.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters}')
    encoders.to('cuda')

    optimizer = create_optimizer(args, filter(lambda p: p.requires_grad, encoders.parameters()))
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    max_acc = {
        'acc_period': 0,
        'acc_side': 0,
        'acc_grade': 0,
        'acc_lesion': 0,
        'acc_p': 0,
    }

    for epoch in range(args.epochs):
        progress_bar = tqdm(total=len(dataset_train) // args.batch_size)
        progress_bar.set_description(f'Epoch {epoch}/{args.epochs}')
        text_encoder.train()
        image_encoder.train()
        text_encoder.text_model.requires_grad_(False)
        image_encoder.vision_model.requires_grad_(False)
        for idx, samples, in enumerate(data_loader_train):
            pixel_values_early = samples['pixel_values_early'].to('cuda', non_blocking=True)
            pixel_values_late = samples['pixel_values_late'].to('cuda', non_blocking=True)
            pixel_values = torch.cat([pixel_values_early, pixel_values_late], dim=0)

            text_early = samples['text_early']
            text_late = samples['text_late']
            prompts = text_early + text_late

            with torch.cuda.amp.autocast():
                text_embeds = run_text_embeds(prompts,
                                              tokenizer,
                                              text_encoder,
                                              text_projection)
                image_embeds, image_embeds_pooling = run_image_embeds(pixel_values,
                                                                      image_encoder,
                                                                      visual_projection)

            loss_identify = clip_loss(
                image_embeds[: args.batch_size],
                image_embeds[args.batch_size:],
                logit_scale)
            loss_multi = clip_loss(
                text_embeds,
                image_embeds_pooling,
                logit_scale)
            loss = loss_identify * 0.01 + loss_multi

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=image_encoder.parameters(), create_graph=is_second_order)
            logs = {
                'loss_multi': loss_multi.detach().item(),
                'loss_identify': loss_identify.detach().item(),
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

        progress_bar.close()
        lr_scheduler.step(epoch)

        if (epoch + 1) % 10 == 0:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(text_encoder.state_dict(), os.path.join(args.output_dir, f'text_checkpoint_{args.fold}.pth'))
            torch.save(image_encoder.state_dict(), os.path.join(args.output_dir, f'image_checkpoint_{args.fold}.pth'))
            acc = eval(args, dataset_test, clip_model, text_encoder, image_encoder)
            max_acc = {
                'acc_side': max(max_acc['acc_side'], acc['acc_side']),
                'acc_period': max(max_acc['acc_period'], acc['acc_period']),
                'acc_grade': max(max_acc['acc_grade'], acc['acc_grade']),
                'acc_lesion': max(max_acc['acc_lesion'], acc['acc_lesion']),
                'acc_p': max(max_acc['acc_p'], acc['acc_p']),
            }
            print(f'Eopch {epoch}: Side {acc["acc_side"]:.2f}%, Period {acc["acc_period"]:.2f}%, Grade {acc["acc_grade"]:.2f}%, Lesion {acc["acc_lesion"]:.2f}%, Proliferative {acc["acc_p"]:.2f}%')
            print(f'Max: Side {max_acc["acc_side"]:.2f}%, Period {max_acc["acc_period"]:.2f}%, Grade {max_acc["acc_grade"]:.2f}%, Lesion {max_acc["acc_lesion"]:.2f}%, Proliferative {max_acc["acc_p"]:.2f}%')


@torch.no_grad()
def eval(args, dataset_test=None, clip_model=None, text_encoder=None, image_encoder=None):
    if dataset_test is None:
        dataset_test = UWFADataset(
            args.dataset_dir,
            args.model_id_org,
            fold=args.fold,
            split='test',
            is_train=False
        )
    sampler = SequentialSampler(dataset_test)
    data_loader = DataLoader(
        dataset_test,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if clip_model is None:
        clip_model = CLIPModel.from_pretrained(args.model_id_org).requires_grad_(False).to('cuda')
        clip_model.eval()

    visual_projection = clip_model.visual_projection
    text_projection = clip_model.text_projection
    logit_scale = clip_model.logit_scale.exp()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_org)

    if text_encoder is None:
        text_encoder = CLIPTextModelPromptTuning.from_pretrained(args.model_id_org)
        text_encoder_checkpoint = torch.load(os.path.join(args.output_dir, f'text_checkpoint_{args.fold}.pth'))
        text_encoder.load_state_dict(text_encoder_checkpoint)
    if image_encoder is None:
        image_encoder = CLIPVisionModelPromptTuning.from_pretrained(args.model_id_org)
        image_encoder_checkpoint = torch.load(os.path.join(args.output_dir, f'image_checkpoint_{args.fold}.pth'))
        image_encoder.load_state_dict(image_encoder_checkpoint)

    text_encoder.to('cuda').eval()
    image_encoder.to('cuda').eval()

    acc_period_list = []
    acc_side_list = []
    acc_grade_list = []
    acc_lesion_list = []
    acc_p_list = []

    side_prompts = [
        'a ffa of a left fundus',
        'a ffa of a right fundus']
    side_embeds = run_text_embeds(side_prompts, tokenizer, text_encoder, text_projection)
    period_prompts = [
        'an early ffa of a fundus',
        'a late ffa of a fundus']
    period_embeds = run_text_embeds(period_prompts, tokenizer, text_encoder, text_projection)
    grade_prompts = [
        'a ffa of a normal fundus',
        'a ffa of a npdr fundus',
        'a ffa of a pdr fundus']
    grade_embeds = run_text_embeds(grade_prompts, tokenizer, text_encoder, text_projection)
    lesion_prompts = [
        'a ffa of a normal fundus',
        'a ffa of a dr fundus']
    lesion_embeds = run_text_embeds(lesion_prompts, tokenizer, text_encoder, text_projection)
    p_prompts = [
        'a ffa of a np fundus',
        'a ffa of a pdr fundus']
    p_embeds = run_text_embeds(p_prompts, tokenizer, text_encoder, text_projection)

    side_embeds = side_embeds / side_embeds.norm(p=2, dim=-1, keepdim=True)
    period_embeds = period_embeds / period_embeds.norm(p=2, dim=-1, keepdim=True)
    grade_embeds = grade_embeds / grade_embeds.norm(p=2, dim=-1, keepdim=True)
    lesion_embeds = lesion_embeds / lesion_embeds.norm(p=2, dim=-1, keepdim=True)
    p_embeds = p_embeds / p_embeds.norm(p=2, dim=-1, keepdim=True)

    for idx, samples, in enumerate(data_loader):
        label_side = torch.cat([samples['side']] * 2, dim=0).to('cuda', non_blocking=True)
        pixel_values_early = samples['pixel_values_early'].to('cuda', non_blocking=True)
        pixel_values_late = samples['pixel_values_late'].to('cuda', non_blocking=True)
        label_grade = torch.cat([samples['grade']] * 2, dim=0).to('cuda', non_blocking=True)
        label_lesion = torch.where(label_grade > 0, 1, 0)
        label_p = torch.where(label_grade == 2, 1, 0)
        label_period = torch.cat([torch.zeros(1, dtype=torch.int64),
                                  torch.ones(1, dtype=torch.int64)], dim=0).to('cuda', non_blocking=True)

        image_embeds = image_encoder(torch.cat([pixel_values_early, pixel_values_late], dim=0))
        image_embeds = image_embeds[1]
        image_embeds = visual_projection(image_embeds)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_side = torch.matmul(image_embeds, side_embeds.t()) * logit_scale
        acc_side = accuracy(logits_side, label_side)[0]
        logits_period = torch.matmul(image_embeds, period_embeds.t()) * logit_scale
        acc_period = accuracy(logits_period, label_period)[0]
        logits_grade = torch.matmul(image_embeds, grade_embeds.t()) * logit_scale
        acc_grade = accuracy(logits_grade, label_grade)[0]
        logits_lesion = torch.matmul(image_embeds, lesion_embeds.t()) * logit_scale
        acc_lesion = accuracy(logits_lesion, label_lesion)[0]
        logits_p = torch.matmul(image_embeds, p_embeds.t()) * logit_scale
        acc_p = accuracy(logits_p, label_p)[0]

        acc_side_list.append(acc_side)
        acc_period_list.append(acc_period)
        acc_grade_list.append(acc_grade)
        acc_lesion_list.append(acc_lesion)
        acc_p_list.append(acc_p)

    acc = {
        'acc_side': sum(acc_side_list) / len(dataset_test),
        'acc_period': sum(acc_period_list) / len(dataset_test),
        'acc_grade': sum(acc_grade_list) / len(dataset_test),
        'acc_lesion': sum(acc_lesion_list) / len(dataset_test),
        'acc_p': sum(acc_p_list) / len(dataset_test),
    }
    return acc


@torch.no_grad()
def save_embeds(args):
    clip_model = CLIPModel.from_pretrained(args.model_id_org).requires_grad_(False).to('cuda')
    clip_model.eval()
    image_encoder = CLIPVisionModelPromptTuning.from_pretrained(args.model_id_org)
    image_encoder_checkpoint = torch.load(os.path.join(args.output_dir, f'image_checkpoint_{args.fold}.pth'))
    image_encoder.load_state_dict(image_encoder_checkpoint)
    image_encoder.to('cuda').eval()

    dataset_test = UWFADataset(
        args.dataset_dir,
        args.model_id_org,
        fold=args.fold,
        split='all',
        is_train=False
    )
    sampler = SequentialSampler(dataset_test)
    data_loader = DataLoader(
        dataset_test,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    for idx, samples, in enumerate(data_loader):
        pixel_values_org = samples['pixel_values_org'].to('cuda', non_blocking=True)
        image_embeds_org = image_encoder(pixel_values_org)[0].squeeze(0)
        torch.save(image_embeds_org, os.path.join(args.dataset_dir, samples['embed'][0]))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)
    save_embeds(args)

