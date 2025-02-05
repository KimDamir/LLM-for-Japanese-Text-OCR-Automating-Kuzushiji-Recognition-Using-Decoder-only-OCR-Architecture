import os
from pathlib import Path

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
)

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))


def story_data():
    return GeneratorCfg(
        num_image=100000,
        save_dir='../../examples/preprocessed_data/text_renderer',
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=32,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            corpus=CharCorpus(
                CharCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "corpus.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(20, 30),
                    length=(1,2),
                    chars_file=CURRENT_DIR / "corpus" / "charset.txt"
                )
            ),
            corpus_effects=Effects(Line(0.3, thickness=(1, 3))),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
        ),
    )


configs = [story_data()]