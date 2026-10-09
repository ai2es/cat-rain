"""Step 10c: Assemble Fig9_POIs_dynamics_map.pdf from the step 10b map and legend PNGs.

Reproduces the original Google Slides layout (720 x 405 pt page). Positions are in slide units (EMU) read
from the content stream of the original PDF; the symbols legend is cropped to the same clip window.

Usage (from the repo root):
    python paper/scripts/10c_build_fig9.py --fig_dir paper/output/figures
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

W, H = 365760.0, 205740.0  # slide size in EMU (720 x 405 pt)


def box(x0, y_top, w, h):
    """Slide box (top-left origin) -> matplotlib figure fraction [left, bottom, width, height]."""
    return [x0 / W, 1 - (y_top + h) / H, w / W, h / H]


def main(fig_dir):
    fig = plt.figure(figsize=(10, 5.625), facecolor="white")

    # two-panel ZIP choropleth, stretched to the slide box
    ax = fig.add_axes(box(0, 25935.0, 334721.0, 143302.42))
    ax.imshow(Image.open(f"{fig_dir}/zip_choropleth_two_panel_no_legend.png").convert("RGB"), aspect="auto")
    ax.axis("off")

    # percent-change color bar
    ax = fig.add_axes(box(332648.0, 66314.5, 31274.998, 73110.98))
    ax.imshow(Image.open(f"{fig_dir}/legend_percent_change.png").convert("RGB"), aspect="auto")
    ax.axis("off")

    # symbols legend, cropped to the slide's clip window
    img = Image.open(f"{fig_dir}/legend_symbols.png").convert("RGB")
    x_img, y_img_top, w_img, h_img = 316858.9, 165572.05 - 33932.87, 63171.098, 33932.87
    cx0, cx1, cy0, cy1 = 331729.5, 364841.5, 142916.0, 153927.45
    crop = (round((cx0 - x_img) / w_img * img.width), round((cy0 - y_img_top) / h_img * img.height),
            round((cx1 - x_img) / w_img * img.width), round((cy1 - y_img_top) / h_img * img.height))
    ax = fig.add_axes(box(cx0, cy0, cx1 - cx0, cy1 - cy0))
    ax.imshow(img.crop(crop), aspect="auto")
    ax.axis("off")

    fig.savefig(f"{fig_dir}/Fig9_POIs_dynamics_map.pdf", dpi=400)
    fig.savefig(f"{fig_dir}/Fig9_POIs_dynamics_map.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig_dir", default="paper/output/figures", help="Directory with the step 10b PNGs")
    main(parser.parse_args().fig_dir)
