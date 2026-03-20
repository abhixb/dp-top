"""Molmo2 client for instruction reward computation.

Based on allenai/Molmo2-8B model from HuggingFace.
Uses Qwen3-8B as base language model with SigLIP 2 vision backbone.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from molmo_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from topreward.clients.qwen import QwenClient
from topreward.metrics.instruction_reward import InstructionRewardResult
from topreward.utils.aliases import Event, ImageEvent, ImageNumpy, ImageT, TextEvent
from topreward.utils.constants import MAX_TOKENS_TO_GENERATE
from topreward.utils.images import to_pil


class Molmo2Client(QwenClient):
    """Client for Molmo2-8B model with instruction reward support."""

    def __init__(
        self,
        model_name: str = "allenai/Molmo2-8B",
        rpm: float = 0.0,
        max_input_length: int = 32768,
    ):
        super(QwenClient, self).__init__(rpm=rpm)

        logger.info(f"Loading Molmo2 model {model_name}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype="bfloat16",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info(f"Molmo2 processor type: {type(self.processor)}")
        self.model_name = model_name
        self.max_input_length = max_input_length

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        """Generate response from provider-agnostic events."""
        messages = [{"role": "user", "content": []}]
        for ev in events:
            if isinstance(ev, TextEvent):
                messages[0]["content"].append({"type": "text", "text": ev.text})
            elif isinstance(ev, ImageEvent):
                messages[0]["content"].append({"type": "image", "image": to_pil(cast(ImageT, ev.image))})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[to_pil(cast(ImageT, ev.image)) for ev in events if isinstance(ev, ImageEvent)],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        input_len = inputs["input_ids"].shape[-1]
        if input_len > self.max_input_length:
            raise ValueError(f"Input length {input_len} exceeds max {self.max_input_length}")
        logger.info(f"Input length: {input_len}")

        # Generate
        self.model.eval()
        with torch.inference_mode():
            if temperature == 0.0:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS_TO_GENERATE,
                    do_sample=False,
                )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS_TO_GENERATE,
                    do_sample=True,
                    temperature=temperature,
                )

        # Decode only the generated part
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def generate_object_state_reasoning(
        self,
        frames: Sequence[ImageT],
        fps: float = 2.0,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a description of the robot manipulation trajectory.

        This generates an instruction-agnostic description to avoid circular dependencies
        where mentioning instruction objects would artificially inflate likelihood scores.

        Args:
            frames: List of images representing the video.
            fps: Frames per second (for API consistency; not directly used in Molmo's
                 timestamp-based format).
            max_new_tokens: Maximum tokens to generate for reasoning.

        Returns:
            Generated text describing the trajectory.
        """
        # Convert frames to PIL images
        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]

        prompt_text = "Describe the robot manipulation trajectory in the video."

        # Create message with video and text (using Molmo's timestamp format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "timestamps": np.arange(len(pil_frames))},
                    # dict(type="image", image=pil_frames[0]),
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Process vision info using molmo_utils
        _, videos, video_kwargs = process_vision_info(messages)
        if videos is None:
            raise ValueError("process_vision_info returned no videos")
        videos, video_metadatas = zip(*videos, strict=False)
        videos, video_metadatas = list(videos), list(video_metadatas)

        # Fix metadata for single-frame edge case
        for idx, _metadata in enumerate(video_metadatas):
            if _metadata["total_num_frames"] == 1:
                video_metadatas[idx]["fps"] = 1.0
                video_metadatas[idx]["frames_indices"] = np.array([1])

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare processor inputs
        inputs = self.processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate description
        self.model.eval()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        # Extract only the generated portion (not prompt echo)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=False)]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()

    def compute_instruction_reward(
        self,
        frames: list[ImageNumpy],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        add_chat_template: bool = False,
    ) -> InstructionRewardResult:
        """Compute a log-likelihood reward for an instruction conditioned on a trajectory of frames.

        This implements the instruction reward approach from "Vision Language Models are
        In-Context Value Learners", measuring how well the trajectory matches the given
        instruction by computing the log-probability of generating the instruction text.

        Args:
            frames: List of images representing the trajectory (at least 2 frames).
            instruction: Instruction text to evaluate.
            reduction: Reduction to apply to token log probabilities ("mean" or "sum").
            fps: Frames per second for video input (default: 2.0).
            use_video_description: If True, generate instruction-agnostic description of
                                  the robot manipulation trajectory, then prepend it as context
                                  before evaluating instruction likelihood. This avoids circular
                                  dependencies that would artificially inflate scores.
            add_chat_template: If True, wrap the full prompt (including instruction) with
                               the chat template before tokenization.

        Returns:
            InstructionRewardResult with the computed reward and metadata.
        """

        pil_frames = [to_pil(cast(ImageT, f)) for f in frames]

        # Optionally generate trajectory description for augmented context
        trajectory_description = None
        if use_video_description:
            logger.info("Generating trajectory description...")
            trajectory_description = self.generate_object_state_reasoning(pil_frames, fps=fps)
            logger.info("Generated trajectory description.")
            logger.info(f"Trajectory description: {trajectory_description}")
            # Prepend trajectory description as object state reasoning
            prompt_text = (
                f"{trajectory_description} Therefore given the above "
                f"description and the video, the video shows a robot "
                f"manipulation trajectory that **completes** the following "
                f"instruction: "
            )
        else:
            # Original prompt without description
            prompt_text = "The above video shows a complete robot manipulation trajectory of the following task with all the motion: "

        full_text = f"{prompt_text}{instruction}\n Decide whether the above statement is True or not. The answer is:"
        if not add_chat_template:
            full_text += " True"

        # process the video and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "timestamps": np.arange(len(pil_frames))},
                    {"type": "text", "text": full_text},
                ],
            }
        ]

        # process the video using `molmo_utils.process_vision_info`
        _, videos, video_kwargs = process_vision_info(messages)
        if videos is None:
            raise ValueError("process_vision_info returned no videos")
        videos, video_metadatas = zip(*videos, strict=False)
        videos, video_metadatas = list(videos), list(video_metadatas)
        for idx, _metadata in enumerate(video_metadatas):
            if _metadata["total_num_frames"] == 1:
                video_metadatas[idx]["fps"] = 1.0
                video_metadatas[idx]["frames_indices"] = np.array([1])

        # apply the chat template to the input messages
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_chat_template),
        )

        eos_token = self.processor.tokenizer.eos_token
        full_text = text.split(eos_token)[0] if not add_chat_template and eos_token is not None else f"{text}True"

        # process the video and text
        inputs = self.processor(
            videos=videos,
            video_metadata=video_metadatas,
            text=full_text,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()

        # Mask the prompt so we only compute loss on the instruction + "True" part
        prompt_length = inputs["input_ids"].shape[1] - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        # generate output
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Compute per-token log probabilities
        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]

        # Apply reduction
        reward = masked_log_probs.sum().item() if reduction == "sum" else masked_log_probs.mean().item()

        # Extract per-token info for metadata
        per_token_log_probs_list = masked_log_probs.detach().cpu().tolist()
        token_ids_list = target_labels[mask].detach().cpu().tolist()

        return InstructionRewardResult(
            reward=reward,
            reduction=reduction,
            token_count=len(per_token_log_probs_list),
            per_token_log_probs=per_token_log_probs_list,
            token_ids=token_ids_list,
            trajectory_description=trajectory_description,
        )
