import os
from pathlib import Path

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    BlackTextColorCfg,
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
)
CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))


def print_effects_data():
    return GeneratorCfg(
        num_image=1000000,
        save_dir='../../examples/preprocessed_data/text_renderer',
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=-1,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            corpus=CharCorpus(
                CharCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "corpus.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(10, 30),
                    length=(1,8),
                    chars_file=CURRENT_DIR / "corpus" / "charset.txt",
                    filter_by_chars=True,
                    filter_font=True,
                    text_color_cfg=SimpleTextColorCfg((200, 255))
                )
            ),
            corpus_effects=Effects(Line(0.15, thickness=(1, 3))),
            gray=False
        ),
    )
    
def color_data():
    return GeneratorCfg(
        num_image=2000000,
        save_dir='../../examples/preprocessed_data/text_renderer',
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=-1,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            corpus=CharCorpus(
                CharCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "corpus.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(10, 30),
                    length=(1,8),
                    chars_file=CURRENT_DIR / "corpus" / "charset.txt",
                    filter_by_chars=True,
                    filter_font=True,
                    text_color_cfg=SimpleTextColorCfg((150, 255))
                )
            ),
            gray=False
        ),
    )
    
def print_data():
    return GeneratorCfg(
        num_image=2000000,
        save_dir='../../examples/preprocessed_data/text_renderer_print',
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=-1,
            corpus=CharCorpus(
                CharCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "corpus.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(10, 30),
                    length=(1,8),
                    chars_file=CURRENT_DIR / "corpus" / "charset.txt",
                    filter_by_chars=True,
                    filter_font=True,
                    text_color_cfg=BlackTextColorCfg()
                )
            ),
            gray=False
        ),
    )
    
def vertical_data():
    return GeneratorCfg(
        num_image=2000000,
        save_dir='../../examples/preprocessed_data/text_renderer_vertical',
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=-1,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            corpus=CharCorpus(
                CharCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "corpus.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(10, 30),
                    horizontal=False,
                    length=(1,8),
                    chars_file=CURRENT_DIR / "corpus" / "charset.txt",
                    filter_by_chars=True,
                    filter_font=True,
                    text_color_cfg=SimpleTextColorCfg((200, 255))
                )
            ),
            gray=False
        ),
    )



configs = [color_data()]