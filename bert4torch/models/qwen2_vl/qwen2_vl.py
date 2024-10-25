from typing import List, Optional, Tuple, Union
from bert4torch.models.qwen import Qwen2
from bert4torch.snippets import DottableDict, inference_mode
import torch


class Qwen2VL(Qwen2):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = DottableDict(config)
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
        from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
        vision_config = Qwen2VLVisionConfig.from_dict(self.config.vision_config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(vision_config, attn_implementation=self.config._attn_implementation)

    def get_vllm_embedding(
            self, 
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            **kwargs
        ):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
            return inputs_embeds, attention_mask

    def apply_embeddings(self, *inputs:Union[tuple, list], **model_kwargs):
        # 准备进embedding层的一些输入
        _, _, _, _, _, attention_mask, model_kwargs = self.preprare_embeddings_inputs(*inputs, **model_kwargs)
        model_kwargs['attention_mask'] = attention_mask
        inputs_embeds, attention_mask = self.get_vllm_embedding(**model_kwargs)

        # 进入embedding层
        model_kwargs.update({'hidden_states': inputs_embeds, 'inputs_embeds': inputs_embeds, 'attention_mask':attention_mask})
        
        return model_kwargs

    def load_variable(self, variable, old_key, new_key):
        if old_key in {'embeddings.word_embeddings.weight', 'lm_head.weight'}:
            return self.load_embeddings(variable)
        return variable
    
    def variable_mapping(self):
        # 映射到权重格式
        mapping = {
            'embeddings.word_embeddings.weight': 'model.embed_tokens.weight',
            'lm_head.weight': 'lm_head.weight',
            'LayerNormFinal.weight': 'model.norm.weight',
            }

        for i in range(self.num_hidden_layers):
            mapping.update( 
            {
            f'decoderLayer.{i}.multiHeadAttention.q.weight': f'model.layers.{i}.self_attn.q_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.q.bias': f'model.layers.{i}.self_attn.q_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.k.weight': f'model.layers.{i}.self_attn.k_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.k.bias': f'model.layers.{i}.self_attn.k_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.v.weight': f'model.layers.{i}.self_attn.v_proj.weight',
            f'decoderLayer.{i}.multiHeadAttention.v.bias': f'model.layers.{i}.self_attn.v_proj.bias',
            f'decoderLayer.{i}.multiHeadAttention.o.weight': f'model.layers.{i}.self_attn.o_proj.weight',
            f'decoderLayer.{i}.attnLayerNorm.weight': f'model.layers.{i}.input_layernorm.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense.weight': f'model.layers.{i}.mlp.gate_proj.weight',
            f'decoderLayer.{i}.feedForward.intermediateDense2.weight': f'model.layers.{i}.mlp.up_proj.weight',
            f'decoderLayer.{i}.feedForward.outputDense.weight': f'model.layers.{i}.mlp.down_proj.weight',
            f'decoderLayer.{i}.ffnLayerNorm.weight': f'model.layers.{i}.post_attention_layernorm.weight'
            })
        return mapping

    
    def _decode_stream(self, inputs_embeds, attention_mask, **kwargs):
        for output in self.model.stream_generate(
            inputs_embeds, attention_mask=attention_mask, **kwargs):
            yield output

    def get_rope_index(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

            Explanation:
                Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

                For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
                Examples:
                    input_ids: [T T T T T], here T is for text.
                    temporal position_ids: [0, 1, 2, 3, 4]
                    height position_ids: [0, 1, 2, 3, 4]
                    width position_ids: [0, 1, 2, 3, 4]

                For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
                and 1D rotary position embeddin for text part.
                Examples:
                    Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                    input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                    vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                    vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                    vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                    text temporal position_ids: [3, 4, 5, 6, 7]
                    text height position_ids: [3, 4, 5, 6, 7]
                    text width position_ids: [3, 4, 5, 6, 7]
                    Here we calculate the text start position_ids as the max vision position_ids plus 1.

            Args:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                    it.
                image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                    The temporal, height and width of feature shape of each image in LLM.
                video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                    The temporal, height and width of feature shape of each video in LLM.
                attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                    - 1 for tokens that are **not masked**,
                    - 0 for tokens that are **masked**.

            Returns:
                position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
                mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
            """
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            image_token_id = self.config.image_token_id
            video_token_id = self.config.video_token_id
            vision_start_token_id = self.config.vision_start_token_id
            mrope_position_deltas = []
            if image_grid_thw is not None or video_grid_thw is not None:
                total_input_ids = input_ids
                position_ids = torch.ones(
                    3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
                )
                image_index, video_index = 0, 0
                for i, input_ids in enumerate(total_input_ids):
                    if attention_mask is not None:
                        input_ids = input_ids[attention_mask[i] == 1]
                    image_nums, video_nums = 0, 0
                    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (vision_tokens == video_token_id).sum()
                    input_tokens = input_ids.tolist()
                    llm_pos_ids_list: list = []
                    st = 0
                    remain_images, remain_videos = image_nums, video_nums
                    for _ in range(image_nums + video_nums):
                        if image_token_id in input_tokens and remain_images > 0:
                            ed_image = input_tokens.index(image_token_id, st)
                        else:
                            ed_image = len(input_tokens) + 1
                        if video_token_id in input_tokens and remain_videos > 0:
                            ed_video = input_tokens.index(video_token_id, st)
                        else:
                            ed_video = len(input_tokens) + 1
                        if ed_image < ed_video:
                            t, h, w = (
                                image_grid_thw[image_index][0],
                                image_grid_thw[image_index][1],
                                image_grid_thw[image_index][2],
                            )
                            image_index += 1
                            remain_images -= 1
                            ed = ed_image
                        else:
                            t, h, w = (
                                video_grid_thw[video_index][0],
                                video_grid_thw[video_index][1],
                                video_grid_thw[video_index][2],
                            )
                            video_index += 1
                            remain_videos -= 1
                            ed = ed_video
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                    if st < len(input_tokens):
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
                return position_ids, mrope_position_deltas
            else:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                    mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
                else:
                    position_ids = (
                        torch.arange(input_ids.shape[1], device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(3, input_ids.shape[0], -1)
                    )
                    mrope_position_deltas = torch.zeros(
                        [input_ids.shape[0], 1],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )

                return position_ids, mrope_position_deltas

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if cache_position is None:
            cache_position = self._get_initial_cache_position(input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values)

        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            kwargs.update({"inputs_embeds": inputs_embeds, "input_ids": None})
        else:
            kwargs.update({"input_ids": input_ids, "inputs_embeds": None})

        kwargs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return kwargs

    def _get_initial_cache_position(self, input_ids, **model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if model_kwargs.get("inputs_embeds") is not None:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = cache[0][0].shape[2]

            cache_position = cache_position[past_length:]

        return cache_position
    
    # @inference_mode()
    # def generate(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     rope_deltas: Optional[torch.LongTensor] = None,
    #     **kwargs
    # ):
    #     cache_position = self._get_initial_cache_position(input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values)
    #     model_inputs = self.prepare_inputs_for_generation(input_ids,
    #             past_key_values=past_key_values,
    #             attention_mask=attention_mask,
    #             inputs_embeds=inputs_embeds,
    #             cache_position=cache_position,
    #             position_ids=position_ids,
    #             use_cache=True,
    #             pixel_values=pixel_values,
    #             pixel_values_videos=pixel_values_videos,
    #             image_grid_thw=image_grid_thw,
    #             video_grid_thw=video_grid_thw,
    #     )
    #     inputs_embeds, attention_mask = self.get_vllm_embedding(**model_inputs)
    #     position_ids = model_inputs['position_ids']
    #     rope_deltas = model_inputs['rope_deltas']
    #     return super().generate(inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
