from dataclasses import dataclass

from topreward.utils.aliases import ImageNumpy
from topreward.utils.errors import OriginalFramesLengthMismatchError, ShuffledFramesIndicesNotSubsetError, ShuffledFramesLengthMismatchError


@dataclass
class Episode:
    """
    Container for a single episode (or a selected subsequence of it) used in
    evaluation/in context learning.

    Attributes
    - instruction: Natural-language description of the task to complete.
    - starting_frame: The first observation of the (sub)episode.
    - episode_index: Index of this episode within the source dataset.
    - original_frames_indices: Sorted indices from the original episode that
        define the selected subsequence.
    - original_frames_task_completion_rates: Per-frame task completion rates for
        the frames referenced by ``original_frames_indices`` (1:1 aligned; i-th
        value corresponds to the i-th index above).
    - shuffled_frames_indices: Indices from the original episode corresponding to
        ``shuffled_frames``, ordered as they are fed to the model (shuffled order).
        Each entry should also exist in ``original_frames_indices``.
    - shuffled_frames: Frames arranged according to ``shuffled_frames_indices``.
    - shuffled_frames_approx_completion_rates: Per-shuffled-frame approximate
        completion rates (1:1 aligned with ``shuffled_frames``).

    Invariants
    - len(original_frames_indices) == len(original_frames_task_completion_rates)
    - len(shuffled_frames_indices) == len(shuffled_frames)
        == len(shuffled_frames_approx_completion_rates)
    - All values in ``shuffled_frames_indices`` refer to frames from the same
        episode namespace as ``original_frames_indices``.
    """

    instruction: str
    starting_frame: ImageNumpy | None
    episode_index: int
    original_frames_indices: list[int]  # subsequence of original episode indices, sorted
    shuffled_frames_indices: list[int]  # original-episode indices in model input (shuffled) order
    shuffled_frames_approx_completion_rates: list[int]  # aligned 1:1 with shuffled_frames
    original_frames_task_completion_rates: list[int]  # aligned 1:1 with original_frames_indices
    shuffled_frames: list[ImageNumpy]  # frames ordered per shuffled_frames_indices
    all_frames: list[ImageNumpy] | None = None  # optional: all frames including starting_frame

    def __post_init__(self):
        if len(self.original_frames_indices) != len(self.original_frames_task_completion_rates):
            raise OriginalFramesLengthMismatchError(
                len(self.original_frames_indices),
                len(self.original_frames_task_completion_rates),
            )
        if not (len(self.shuffled_frames_indices) == len(self.shuffled_frames) == len(self.shuffled_frames_approx_completion_rates)):
            raise ShuffledFramesLengthMismatchError(
                len(self.shuffled_frames_indices),
                len(self.shuffled_frames),
                len(self.shuffled_frames_approx_completion_rates),
            )
        # Optional: ensure shuffled indices are a subset of original indices
        if not set(self.shuffled_frames_indices).issubset(set(self.original_frames_indices)):
            raise ShuffledFramesIndicesNotSubsetError()

    def get_uniformly_spaced_frames(self) -> list[ImageNumpy]:
        """Return shuffled_frames reordered to chronological (original) order.

        This unshuffles the frames by sorting them according to their original
        indices. Useful for instruction reward computation where temporal order
        matters.

        Returns:
            List of frames in chronological order (sorted by original frame
            index).
        """
        # Pair each frame with its original index, sort by index, extract frames
        paired = list(zip(self.shuffled_frames_indices, self.shuffled_frames, strict=False))
        paired.sort(key=lambda x: x[0])  # Sort by original index
        return [frame for _, frame in paired]


@dataclass
class InferredEpisode(Episode):
    """
    Extension of Episode that includes model-predicted completion rates for
    the shuffled frames.
    """

    shuffled_frames_predicted_completion_rates: list[int] | None = None  # should be aligned 1:1 with shuffled_frames
    # if not, that means that model failed to predict for all frames
    # (e.g. returned incomplete list of preds)

    @classmethod
    def from_predictions(cls, episode: Episode, predictions: list[int]) -> "InferredEpisode":
        """Simple factory method to create an InferredEpisode from an Episode
        and predictions.
        """
        return cls(
            instruction=episode.instruction,
            starting_frame=episode.starting_frame,
            episode_index=episode.episode_index,
            original_frames_indices=episode.original_frames_indices,
            shuffled_frames_indices=episode.shuffled_frames_indices,
            shuffled_frames_approx_completion_rates=(episode.shuffled_frames_approx_completion_rates),
            original_frames_task_completion_rates=(episode.original_frames_task_completion_rates),
            shuffled_frames=episode.shuffled_frames,
            shuffled_frames_predicted_completion_rates=predictions,
        )


@dataclass
class Example:
    """
    Container for a single training/evaluation example consisting of one
    evaluation episode and multiple context episodes.
    """

    eval_episode: Episode
    context_episodes: list[Episode]

    def __repr__(self) -> str:
        eval_frames = len(self.eval_episode.shuffled_frames)
        ctx_count = len(self.context_episodes)
        ctx_frames_list = [len(ep.shuffled_frames) for ep in self.context_episodes]
        ctx_frames_total = sum(ctx_frames_list)
        return (
            "Example("
            f"eval_episode_index={self.eval_episode.episode_index}, "
            f"eval_frames={eval_frames}, "
            f"context_episodes={ctx_count}, "
            f"context_frames_per_episode={ctx_frames_list}, "
            f"context_frames_total={ctx_frames_total}"
            ")"
        )


@dataclass
class InferredFewShotResult:
    """
    Container for a single evaluation example consisting of one
    evaluation episode and multiple context episodes, with model predictions.
    """

    eval_episode: InferredEpisode
    context_episodes: list[Episode]
