from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class BasePlotter:
    def __init__(
        self,
        output_dir: str | Path = "artifacts/figures",
        dpi: int = 300,
        figsize: tuple = (8, 6),
        style: str = "default",
    ):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dpi = dpi
        self.figsize = figsize
        self.style = style

        plt.style.use(self.style)

    def create_figure(
        self,
        figsize=None,
        ):
        if figsize is None:
            figsize = self.figsize

        fig, ax = plt.subplots(
            figsize=figsize
        )
        return fig, ax

    def create_subplots(
        self,
        nrows=1,
        ncols=1,
        figsize=None,
        **kwargs,
    ):
        if figsize is None:
            figsize = self.figsize
        fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        **kwargs,
    )
        return fig, axes
    
    def style_axis(
        self,
        ax,
        title: str,
        xlabel: str,
        ylabel: str,
        grid: bool = True,
    ):

        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
        )

        ax.set_xlabel(
            xlabel,
            fontsize=12,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=12,
        )

        if grid:
            ax.grid(
                alpha=0.3,
                linestyle="--",
            )

    def save_figure(
        self,
        fig,
        filename: str,
        close: bool = True,
    ):

        filepath = self.output_dir / filename

        fig.tight_layout()

        fig.savefig(
            filepath,
            dpi=self.dpi,
            bbox_inches="tight",
        )

        logger.info(
            "Figure saved -> %s",
            filepath.resolve(),
        )

        if close:
            plt.close(fig)

        return filepath

    def show(
        self,
        fig,
    ):
        plt.show()

    def finish(
        self,
        fig,
        filename,
        save=True,
        show=False,
    ):
        path = None
        if save:
            path = self.save_figure(
            fig,
            filename,
        )
        if show:
            self.show(fig)

        return path