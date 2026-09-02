"""Content modes: live action and animation, chosen by the user.

FluxCutter has two pipelines because one set of models does not cover both
kinds of footage. Measured on this project's own test files: the live-action
detector finds 0.26 faces per sampled frame on `animation.mp4` and most of
what it finds is scenery, while the animation detector finds 0.45 and the
montages show it catching the actual characters.

The mode is always the user's choice. Nothing here inspects the video to
guess which pipeline to run -- not detection counts, not confidence, not
colour statistics. A wrong guess would silently produce a plausible-looking
gallery of the wrong thing, and the user cannot audit a guess they were
never told about.

What a mode owns
----------------
A mode is not just a pair of models. It carries its own detection settings,
its own grouping thresholds and its own tracker floor, because none of those
numbers transfer. The clearest case is the similarity scale: two *different*
anime characters score a median 0.567 under CCIP, where two different people
score 0.03 under ArcFace. Running animation footage against the live-action
floor of 0.35 would merge the entire cast into one person.

That is also why every mode names an embedding space, and why the grouper
refuses to compare across them (see grouper.MixedEmbeddingSpaces).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.models import MODELS, ModelSpec, find_model

LIVE = "live"
ANIMATION = "animation"


@dataclass(frozen=True)
class DetectionDefaults:
    """Detection settings for one mode. Never shared between modes."""

    confidence_threshold: float
    min_face_size: int
    # The grouper's own floor on detector confidence, which is a second,
    # stricter gate than the detector's. It has to be per mode because the
    # two detectors do not score alike: live action's 0.7 would discard most
    # of what the animation detector finds, which peaks around 0.88 and does
    # useful work down at 0.3.
    min_confidence: float


@dataclass(frozen=True)
class GroupingDefaults:
    """Grouping thresholds for one mode.

    These are properties of an embedding model's similarity distribution, so
    they are stored per mode rather than as module constants that one mode
    would silently impose on the other.
    """

    similarity_threshold: float
    consolidation_threshold: float
    contradiction_floor: float
    min_group_eye_span: float


@dataclass(frozen=True)
class ModeSpec:
    """One content mode: what it is called, what it loads, how it is tuned."""

    id: str
    display_name: str
    summary: str
    embedding_space: str
    detector_model: ModelSpec
    embedder_model: ModelSpec
    detection: DetectionDefaults
    grouping: GroupingDefaults
    # Imported lazily inside the factories: animation pulls in onnxruntime,
    # and a user who only ever runs live action should not need it installed
    # -- nor should importing this module cost the load of either backend.
    build_detector: Callable[..., object] = field(repr=False, default=None)
    build_embedder: Callable[..., object] = field(repr=False, default=None)
    extra_requirements: tuple[str, ...] = ()

    @property
    def weights(self) -> tuple[ModelSpec, ModelSpec]:
        return (self.detector_model, self.embedder_model)

    @property
    def download_megabytes(self) -> float:
        """How much this mode would download on a machine that has nothing."""
        return sum(
            spec.size_bytes for spec in self.weights if find_model(spec) is None
        ) / 1_000_000


# --------------------------------------------------------------- factories
#
# Each returns a fresh backend. They are functions rather than classes held
# on the spec so that importing this module imports neither backend: loading
# every model at startup is exactly what a machine with 8 GB cannot afford,
# and animation's two files are 195 MB before onnxruntime's own allocations.


def _live_detector(confidence_threshold: float, **_ignored):
    from app.faces.detector import FaceDetector

    return FaceDetector(confidence_threshold=confidence_threshold)


def _live_embedder(**_ignored):
    from app.faces.embedder import FaceEmbedder

    return FaceEmbedder()


def _anime_detector(confidence_threshold: float, min_face_size: int = 24, **_ignored):
    from app.faces.anime import AnimeDetectorSettings, AnimeFaceDetector

    return AnimeFaceDetector(
        settings=AnimeDetectorSettings(
            confidence_threshold=confidence_threshold, min_face_size=min_face_size
        )
    )


def _anime_embedder(**_ignored):
    from app.faces.anime import AnimeFaceEmbedder

    return AnimeFaceEmbedder()


MODES: dict[str, ModeSpec] = {
    LIVE: ModeSpec(
        id=LIVE,
        display_name="Live Action",
        summary=(
            "YuNet detection with ArcFace identity embeddings. The original "
            "FluxCutter pipeline, tuned on filmed footage."
        ),
        embedding_space="arcface-w600k-r50",
        detector_model=MODELS["detector"],
        embedder_model=MODELS["embedder"],
        detection=DetectionDefaults(
            confidence_threshold=0.6, min_face_size=40, min_confidence=0.7
        ),
        # Unchanged from the values the accuracy work settled on; this mode
        # is the existing pipeline and must behave exactly as it did.
        grouping=GroupingDefaults(
            similarity_threshold=0.35,
            consolidation_threshold=0.375,
            contradiction_floor=0.25,
            min_group_eye_span=0.15,
        ),
        build_detector=_live_detector,
        build_embedder=_live_embedder,
    ),
    ANIMATION: ModeSpec(
        id=ANIMATION,
        display_name="Animation",
        summary=(
            "Anime-trained face detection with CCIP character embeddings. "
            "For drawn footage, where the live-action models mostly find "
            "scenery."
        ),
        embedding_space="ccip-caformer-24",
        detector_model=MODELS["anime_detector"],
        embedder_model=MODELS["anime_embedder"],
        # Lower than live action because this detector scores on its own
        # scale, and because drawn faces at a distance are still legible
        # where a filmed face that small would not be.
        detection=DetectionDefaults(
            confidence_threshold=0.30, min_face_size=24, min_confidence=0.30
        ),
        # Provisional. Measured on 20 labelled character crops from
        # animation.mp4: same-character pairs bottom out at 0.685 (p5 0.745)
        # and different-character pairs reach p95 0.763. There is real
        # overlap, so no threshold is clean here; 0.75 is the balance point
        # measured, and it is a far smaller sample than the live-action
        # numbers rest on. See Instructions.md 17.
        grouping=GroupingDefaults(
            similarity_threshold=0.75,
            consolidation_threshold=0.85,
            contradiction_floor=0.60,
            # The detector returns no landmarks, so the non-face filter has
            # nothing to measure. Set to zero to say that plainly rather
            # than leave a live-action number sitting inert.
            min_group_eye_span=0.0,
        ),
        build_detector=_anime_detector,
        build_embedder=_anime_embedder,
        extra_requirements=("onnxruntime",),
    ),
}

DEFAULT_MODE = LIVE


@dataclass(frozen=True)
class ModeAvailability:
    """Whether a mode can run here, and what is missing if not."""

    mode_id: str
    runtime_ready: bool
    missing_requirements: tuple[str, ...]
    missing_weights: tuple[ModelSpec, ...]

    @property
    def usable(self) -> bool:
        """Runnable now, without downloading or installing anything."""
        return self.runtime_ready and not self.missing_weights

    @property
    def installable(self) -> bool:
        """Runnable once the weights are fetched, with nothing else to install."""
        return self.runtime_ready

    @property
    def download_megabytes(self) -> float:
        return sum(spec.size_bytes for spec in self.missing_weights) / 1_000_000

    def describe(self) -> str:
        """A line a person can act on."""
        if self.usable:
            return "Ready"
        if not self.runtime_ready:
            missing = " ".join(self.missing_requirements)
            return f"Needs {missing} — pip install {missing}"
        names = ", ".join(spec.description for spec in self.missing_weights)
        return f"Needs a {self.download_megabytes:.0f} MB download ({names})"


def missing_requirements(spec: ModeSpec) -> tuple[str, ...]:
    """Which of a mode's optional packages are not importable."""
    import importlib.util

    return tuple(
        name
        for name in spec.extra_requirements
        if importlib.util.find_spec(name) is None
    )


def availability(mode_id: str) -> ModeAvailability:
    """What stands between this machine and running `mode_id`."""
    spec = get_mode(mode_id)
    return ModeAvailability(
        mode_id=spec.id,
        runtime_ready=not missing_requirements(spec),
        missing_requirements=missing_requirements(spec),
        missing_weights=tuple(s for s in spec.weights if find_model(s) is None),
    )


def get_mode(mode_id: str) -> ModeSpec:
    """The named mode.

    Raises:
        KeyError: With the valid names in the message, because this is
            reachable from a command line and a bare KeyError is not a
            usable error report.
    """
    try:
        return MODES[mode_id]
    except KeyError:
        raise KeyError(
            f"Unknown content mode {mode_id!r}. Choose one of: "
            f"{', '.join(sorted(MODES))}."
        ) from None


def mode_ids() -> list[str]:
    """Every mode id, live action first."""
    return [LIVE, ANIMATION]
