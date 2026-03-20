import os
import secrets
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from google.genai.client import Client
from google.genai.types import Blob, GenerateContentConfig, Part
from loguru import logger

from topreward.clients.base import BaseModelClient
from topreward.metrics.instruction_reward import InstructionRewardResult
from topreward.utils.aliases import Event, ImageEvent, ImageNumpy, ImageT, TextEvent
from topreward.utils.images import encode_image


class GeminiClient(BaseModelClient):
    """Gemini client with Vertex AI support and instruction reward computation."""

    @staticmethod
    def _extract_http_status(exc: BaseException) -> int | None:
        """Best-effort extraction of an HTTP status code from various client exceptions."""
        for attr in ("status_code", "status", "code"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val

        resp = getattr(exc, "response", None)
        if resp is not None:
            for attr in ("status_code", "status"):
                val = getattr(resp, attr, None)
                if isinstance(val, int):
                    return val

        # Some libraries stash status under args or nested dicts; keep this conservative.
        return None

    @classmethod
    def _is_transient_error(cls, exc: BaseException) -> bool:
        status = cls._extract_http_status(exc)
        if status in (429, 500, 502, 503, 504):
            return True

        # Fallback heuristics for SDK/network errors that don't expose status codes.
        msg = str(exc).lower()
        if any(s in msg for s in ("503", "service unavailable", "temporarily unavailable", "deadline exceeded", "connection reset")):
            return True

        return isinstance(exc, (TimeoutError, ConnectionError))

    def _generate_content_with_retry(self, *, contents: list, config: GenerateContentConfig):
        """Call Gemini generate_content with retry/backoff for transient errors (e.g., 503)."""
        # Keep these defaults conservative to avoid long stalls.
        max_attempts = 8
        base_sleep_s = 1.0
        max_sleep_s = 30.0

        # Apply rate limiting once per request (not per retry)
        if self._rate_limiter is not None:
            with self._rate_limiter:
                return self._generate_content_with_retry_impl(contents, config, max_attempts, base_sleep_s, max_sleep_s)
        else:
            return self._generate_content_with_retry_impl(contents, config, max_attempts, base_sleep_s, max_sleep_s)

    def _generate_content_with_retry_impl(self, contents: list, config: GenerateContentConfig, max_attempts: int, base_sleep_s: float, max_sleep_s: float):
        """Implementation of retry logic without rate limiting."""
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except BaseException as e:
                last_exc = e
                if attempt >= max_attempts or not self._is_transient_error(e):
                    raise

                # Exponential backoff with jitter.
                sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
                jitter = 0.8 + 0.4 * (secrets.randbelow(10_000) / 10_000)
                sleep_s = sleep_s * jitter
                status = self._extract_http_status(e)
                logger.warning(
                    "Gemini request failed (attempt {}/{}{}, retrying in {:.1f}s): {}",
                    attempt,
                    max_attempts,
                    f", status={status}" if status is not None else "",
                    sleep_s,
                    repr(e),
                )
                time.sleep(sleep_s)

        # Should be unreachable, but keeps type-checkers happy.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini generate_content failed with unknown error")

    def __init__(
        self,
        *,
        rpm: float = 0.0,
        model_name: str,
        project_id: str | None = None,
        location: str = "us-central1",
        use_vertex_ai: bool = True,
    ):
        super().__init__(rpm=rpm)
        self.model_name = model_name
        self.use_vertex_ai = use_vertex_ai

        if use_vertex_ai:
            # Use Vertex AI with better logprobs support
            project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise OSError("GOOGLE_CLOUD_PROJECT environment variable must be set for Vertex AI")
            self.client = Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            logger.info(f"Using Gemini model {self.model_name} via Vertex AI (project: {project_id}, location: {location})")
        else:
            # Use standard Gemini API
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise OSError("GEMINI_API_KEY environment variable must be set")
            self.client = Client(api_key=api_key)
            logger.info(f"Using Gemini model {self.model_name} via API")

    @staticmethod
    def _frames_to_video_bytes(frames: list[ImageNumpy], fps: float = 2.0) -> bytes:
        """Convert a list of frames to MP4 video bytes.

        Args:
            frames: List of image frames as numpy arrays (H, W, C) in RGB format.
            fps: Frames per second for the output video.

        Returns:
            MP4 video as bytes.
        """
        if not frames:
            raise ValueError("frames list is empty")
        # Create temporary file for video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Normalize first frame to establish correct dimensions
            first_frame = GeminiClient._to_rgb_uint8(frames[0])
            height, width = first_frame.shape[:2]

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError("Failed to initialize video writer")

            # Write frames (convert RGB to BGR for OpenCV)
            for idx, frame in enumerate(frames):
                frame_arr = first_frame if idx == 0 else GeminiClient._to_rgb_uint8(frame)
                # Convert RGB to BGR without calling into OpenCV color conversion
                bgr_frame = frame_arr[:, :, ::-1].copy()
                writer.write(bgr_frame)

            writer.release()

            # Read video bytes
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    @staticmethod
    def _to_rgb_uint8(frame: ImageNumpy) -> np.ndarray:
        """Normalize a frame to RGB uint8 contiguous format."""
        arr = np.asarray(frame)
        # If channel-first (C, H, W), move to H, W, C
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        # Ensure channel dimension exists
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # Drop alpha via PIL to be safe
            from PIL import Image

            arr = np.array(Image.fromarray(arr).convert("RGB"))
        # Cast to uint8
        if arr.dtype != np.uint8:
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        # Make contiguous for OpenCV/video writer safety
        return np.ascontiguousarray(arr)

    def _generate_from_events(self, events: list[Event], temperature: float) -> str:
        contents: list = []
        for ev in events:
            if isinstance(ev, TextEvent):
                contents.append(ev.text)
            elif isinstance(ev, ImageEvent):
                img = cast(ImageT, ev.image)
                contents.append(Part.from_bytes(data=encode_image(img), mime_type="image/png"))

        logger.debug(f"Contents length: {len(contents)} parts")
        response = self._generate_content_with_retry(
            contents=contents,
            config=GenerateContentConfig(temperature=temperature),
        )
        if response.text is None:
            raise RuntimeError("Gemini returned no text in response")
        return response.text

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
            fps: Frames per second for video input (not used by Gemini API).
            max_new_tokens: Maximum tokens to generate for reasoning.

        Returns:
            Generated text describing the trajectory.
        """
        # Convert frames to PIL then encode
        contents = []
        for frame in frames:
            contents.append(Part.from_bytes(data=encode_image(frame), mime_type="image/png"))

        # Add prompt
        contents.append("Describe the robot manipulation trajectory shown in these frames:")

        # Generate description
        response = self._generate_content_with_retry(
            contents=contents,
            config=GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_new_tokens,
            ),
        )

        if response.text is None:
            raise RuntimeError("Gemini returned no text for trajectory description")

        return response.text.strip()

    def compute_instruction_reward(
        self,
        frames: list[ImageNumpy],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        use_video_input: bool = True,
    ) -> InstructionRewardResult:
        """Compute instruction reward using Vertex AI logprobs.

        Note: Vertex AI has limitations compared to local models:
        - Cannot mask prompt tokens like Qwen
        - Returns logprobs only for generated tokens

        Args:
            frames: List of images representing the trajectory (at least 2 frames).
            instruction: Instruction text to evaluate.
            reduction: Reduction to apply to token log probabilities ("mean" or "sum").
            fps: Frames per second for video encoding (used when use_video_input=True).
            use_video_description: If True, generate trajectory description first.
            use_video_input: If True, send frames as video. If False, send as individual images.

        Returns:
            InstructionRewardResult with the computed reward and metadata.
        """

        if not self.use_vertex_ai:
            raise NotImplementedError("Instruction reward requires Vertex AI (use_vertex_ai=True) for logprobs support")

        # Optionally generate trajectory description for augmented context
        trajectory_description = None
        if use_video_description:
            logger.info("Generating trajectory description...")
            trajectory_description = self.generate_object_state_reasoning(frames, fps=fps)
            logger.info("Generated trajectory description.")
            logger.info("Trajectory description: %s", trajectory_description)
            prompt_text = (
                f"{trajectory_description} Therefore given the above "
                f"description and the video, the video shows a robot "
                f"manipulation trajectory that **completes** the following "
                f"instruction: "
            )
        else:
            prompt_text = "The above video shows a robot manipulation trajectory that completes the following task: "

        # Build contents with frames + prompt
        # IMPORTANT: Do NOT include "True" in the prompt - we want to measure the
        # probability of the model GENERATING "True" (not tokens after "True")
        contents = []
        if use_video_input:
            # Send as video for better temporal understanding
            logger.debug(f"Converting {len(frames)} frames to video at {fps} FPS")
            video_bytes = self._frames_to_video_bytes(frames, fps=fps)
            logger.debug(f"Video size: {len(video_bytes) / 1024:.1f} KB")
            contents.append(Part(inline_data=Blob(data=video_bytes, mime_type="video/mp4")))
        else:
            # Send as individual image frames
            logger.debug(f"Sending {len(frames)} individual image frames")
            for frame in frames:
                contents.append(Part.from_bytes(data=encode_image(frame), mime_type="image/png"))

        contents.append(f"{prompt_text}{instruction}. \n\n Decide whether the above statement is True or not. Respond with only 'True' or 'False'.")
        # breakpoint()
        # Call Vertex AI with logprobs enabled
        response = self._generate_content_with_retry(
            contents=contents,
            config=GenerateContentConfig(
                response_logprobs=True,
                logprobs=10,
                temperature=0.0,
                max_output_tokens=10000,  # Limit generation to short response
            ),
        )

        # Extract log probabilities from response
        candidates = response.candidates
        if not candidates:
            raise RuntimeError("Gemini response has no candidates")
        candidate_0 = candidates[0]
        logprobs_result_obj = getattr(candidate_0, "logprobs_result", None)
        if logprobs_result_obj is None:
            raise RuntimeError("Gemini response does not contain logprobs. Ensure model supports logprobs and Vertex AI is configured correctly.")
        logprobs_result = logprobs_result_obj.top_candidates[0].candidates
        token_log_probs: float = -20.0
        for candidate in logprobs_result:
            if candidate.token.strip().lower() == "true":
                token_log_probs = candidate.log_probability
                break
        if token_log_probs == -20.0:
            raise RuntimeError("No log probabilities returned from Vertex AI")

        # Apply reduction
        if reduction != "mean":
            raise ValueError(f"Unknown reduction: {reduction}.")
        reward = token_log_probs

        return InstructionRewardResult(
            reward=reward,
            reduction=reduction,
            token_count=1,  # Only one token ("True")
            per_token_log_probs=[token_log_probs],
            trajectory_description=trajectory_description,
        )

    @staticmethod
    def normalize_rewards(
        rewards: Sequence[float],
        method: str = "minmax",
    ) -> np.ndarray:
        """Normalize a sequence of instruction rewards to a 0-1 range.

        This is useful for comparing rewards across different trajectories or
        trajectory prefixes, as raw log-likelihood values can be hard to interpret.

        Args:
            rewards: Sequence of raw reward values (log-likelihoods).
            method: Normalization method. Options:
                - "minmax": Scale to [0, 1] using min-max normalization.

        Returns:
            Normalized rewards as a numpy array in [0, 1] range.
        """
        rewards_arr = np.array(rewards, dtype=np.float64)

        if len(rewards_arr) == 0:
            return rewards_arr

        if len(rewards_arr) == 1:
            return np.array([1.0])

        if method == "minmax":
            r_min, r_max = rewards_arr.min(), rewards_arr.max()
            if r_max == r_min:
                # All rewards are identical
                return np.ones_like(rewards_arr)
            return (rewards_arr - r_min) / (r_max - r_min)

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax'.")

    def compute_instruction_rewards_for_prefixes(  # type: ignore[override]
        self,
        frames: list[ImageNumpy],
        instruction: str,
        num_samples: int = 15,
        reduction: str = "mean",
        fps: float = 2.0,
        use_video_description: bool = False,
        use_video_input: bool = True,
    ) -> InstructionRewardResult:
        """Compute instruction rewards for trajectory prefixes of varying lengths.

        This is useful for analyzing how reward changes as the trajectory progresses.

        Args:
            frames: Full list of trajectory frames.
            instruction: Instruction text to evaluate.
            num_samples: Number of prefix lengths to sample (uniformly spaced).
            reduction: Reduction method ("mean" or "sum").
            fps: Frames per second for video encoding.
            use_video_description: Whether to generate trajectory description.
            use_video_input: If True, send frames as video. If False, send as individual images.

        Returns:
            InstructionRewardResult with prefix_lengths, prefix_rewards, and normalized_prefix_rewards.
            The main reward is the full trajectory reward (last prefix).
        """

        num_frames = len(frames)
        num_samples = min(num_samples, num_frames)

        # Generate uniformly spaced prefix lengths from 2 to full trajectory
        if num_frames > 2:
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted({int(x) for x in prefix_lengths})
        else:
            prefix_lengths = [num_frames]

        rewards = []
        token_counts = []
        for length in prefix_lengths:
            prefix_frames = frames[:length]
            result = self.compute_instruction_reward(
                frames=prefix_frames,
                instruction=instruction,
                reduction=reduction,
                fps=fps,
                use_video_description=use_video_description,
                use_video_input=use_video_input,
            )
            rewards.append(result.reward)
            token_counts.append(result.token_count)
            logger.info(f"Prefix {length:3d} frames: reward={result.reward:.4f} ({result.token_count} tokens)")

        normalized_rewards = self.normalize_rewards(rewards).tolist()

        # Full trajectory is the last prefix
        return InstructionRewardResult(
            reward=rewards[-1],
            reduction=reduction,
            token_count=token_counts[-1],
            prefix_lengths=list(prefix_lengths),
            prefix_rewards=rewards,
            normalized_prefix_rewards=normalized_rewards,
        )
