import itertools
import os
from typing import Any, Dict, List, Tuple, Union, Optional
import copy 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.graph_objs._figure import Figure as FigureType  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from .types import Grid, GridPairs, JSONTask
from .general import touch_dir, adjust_data_format

colors = [
    "#000000",  # Black for 0
    "#0074D9",  # Blue for 1
    "#FF4136",  # Red for 2
    "#2ECC40",  # Green for 3
    "#FFDC00",  # Yellow for 4
    "#AAAAAA",  # Grey for 5
    "#F012BE",  # Fucsha for 6
    "#FF851B",  # Orange for 7
    "#7FDBFF",  # Teal for 8
    "#870C25",  # Brown for 9
    "#FFC0CB",  # Pink for 10, joker symbol for outer boundaries
    "#FFFFFF",  # White for 11, joker symbol for emptiness
]


class ArcPlotter:
    @staticmethod
    def plot_grid(grid: Grid) -> FigureType:
        plot(grid).show()

    @staticmethod
    def _from_gridpairs_to_gridlias(grid_pairs: GridPairs) -> List[Grid]:
        grids_in: List[Grid] = []
        grids_out: List[Grid] = []
        for grid_pair in grid_pairs:
            grid_in, grid_out = grid_pair  # Explicitly type as Grid
            if isinstance(grid_in, list) and all(
                isinstance(row, list) for row in grid_in
            ):
                grids_in.append(grid_in)  # Safely append after type check
            if isinstance(grid_out, list) and all(
                isinstance(row, list) for row in grid_out
            ):
                grids_out.append(grid_out)
        flat_grid_list: List[Grid] = grids_in + grids_out
        return flat_grid_list

    def plot_grid_pairs(self, grid_pairs: GridPairs, task_name: str = "") -> FigureType:
        grid_list = self._from_gridpairs_to_gridlias(grid_pairs)
        plot_figures_on_canvas(grid_list, task_name).show()

    def plot_grid_list(self, grid_list: List[Grid], task_name: str = "") -> FigureType:
        return plot_figures_on_canvas(grid_list, task_name)

    def plot_task(self, task: JSONTask, task_name: str = "", path: Optional[str]=None) -> FigureType:
        task_ = copy.deepcopy(task)
        fake_pair = {
            "input": [[10, 11, 11], [11, 10, 11], [11, 10, 11]],
            "output": [[11, 10, 11], [11, 10, 11], [11, 11, 10]],
        }
        task_['train'].append(fake_pair)
        training_pair, test_pair = adjust_data_format(
            [task_]
        )  # -> Tuple[Dict[str, List[Tuple[Grid, Grid]]], Dict[str, List[Tuple[Grid, Grid]]]]
        # task = [training_pair[k] + test_pair[k] for k in training_pair.keys()]
        grid_pairs = next(iter(training_pair.values())) + next(iter(test_pair.values()))
        pl = self.plot_grid_list(
            grid_list=[x[0] for x in grid_pairs] + [x[1] for x in grid_pairs],
            task_name=task_name,
        )
        pl.write_image(path) if path is not None else pl.show()


def plot(grid, name: str = "") -> FigureType:
    fig = go.Figure()

    grid_copy = [[col for col in row] for row in grid]
    dcolorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    grid_copy.reverse()
    matrix = np.array(grid_copy)
    # Create heatmap for each matrix
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            colorscale=dcolorscale,
            zmin=0,
            zmax=len(colors) - 1,
            xgap=1,
            ygap=1,
            showscale=False,
            name=name,
        )
    )
    fig.update_layout(
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
    )
    fig.update_layout(
        width=max(matrix.shape[1] * 30 + 100, 300),  # Number of columns for width
        height=max(matrix.shape[0] * 30 + 100, 300),  # Number of rows for height
    )

    return fig


def plot_figures_on_canvas(grid_list: List[Grid], task_name: str = "") -> FigureType:
    """In the input list, First put the n inputs, then put the created outputs"""
    # Determine number of figures and rows (2 rows in this case)
    figures_list = [plot(grid) for grid in grid_list]
    n_figures = len(figures_list)
    n_rows = 2
    n_columns = n_figures // n_rows
    # Create a subplot with 2 rows and n_columns
    fig = make_subplots(rows=n_rows, cols=n_columns)

    # Loop over figures and add them to the correct subplot
    for i, figure in enumerate(figures_list):
        row = (i // n_columns) + 1
        col = (i % n_columns) + 1
        for trace in figure["data"]:
            fig.add_trace(trace, row=row, col=col)
        # Set axis ranges for this subplot based on the grid dimensions
        fig.update_xaxes(range=[-0.5, len(grid_list[i][0]) - 0.5], row=row, col=col)
        fig.update_yaxes(range=[-0.5, len(grid_list[i]) - 0.5], row=row, col=col)

    # Create a custom colorscale to define discrete steps
    colorscale = []
    for i, color in enumerate(colors):
        scale_value = i / len(colors)  # Divide by number of intervals to create steps
        colorscale.append([scale_value, color])
        colorscale.append([(i + 1) / len(colors), color])

    # Define tick values and labels for the colorbar
    tick_vals = [
        i + 0.5 for i in range(len(colors) + 1)
    ]  # Center ticks in the middle of each color
    tick_texts = [str(i) for i in range(len(colors) + 1)]

    # Create a dummy Heatmap for colorbar representation with 10 discrete levels
    colorbar_trace = go.Heatmap(
        z=[[i for i in range(len(colors) + 1)]],  # Dummy data to display all colors
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            title="Color Scale",
            lenmode="pixels",
            len=700,  # Adjust length as needed
            outlinecolor="black",  # Set outline color
            outlinewidth=1,  # Set outline width
        ),
        hoverinfo="skip",
        opacity=0,  # Make the heatmap itself invisible, keeping only the colorbar
    )

    # Add the dummy trace to the figure (for displaying the colorbar)
    fig.add_trace(
        colorbar_trace, row=1, col=n_columns
    )  # Place the colorbar on the last column

    # Update layout and show the figure
    fig.update_layout(height=500 * n_rows, width=500 * n_columns, title_text=task_name)
    return fig


class MatplotlibARCPlot:
    """Namespace for matplotlib-based plotting functions."""

    def show(
        self,
        task: Dict,
        title: str = "ARC task",
        plot_size: int = 2,
        grid_lines_width: float = 0.3,
        grid_lines_color: str = "w",
        taskname: str = "-",
        savefig: bool = False,
        dirsave: str = "./Arc_images/",
    ) -> None:
        """Plot an ARC task.

        :param task: dictionary containing train and test pairs, as read from a JSON
        :param title: title to be added in plot
        :param plot_size: dimension of the plot
        :param grid_lines_width: floating-point width of lines separating pixels in the plot
        :param grid_lines_color: color to be used to visually separate grid cells.
            Matplotlib needs to recognise this as a valid color string, so for example
            the string "w" can be passed for white, and the string "k" can be passed for black.
        """
        assert "train" in task
        assert "test" in task

        self.taskname = taskname
        self.savefig = savefig
        self.dirsave = dirsave

        # Note: the `fake_pair` is just used to visually separate the
        # train and test pairs
        fake_pair = {
            "input": [[0, 10, 10], [10, 0, 10], [10, 0, 10]],
            "output": [[10, 0, 10], [10, 0, 10], [10, 10, 0]],
        }
        all_pairs = task["train"] + [fake_pair] + task["test"]
        self.plot_pairs(
            all_pairs,
            title,
            plot_size=plot_size,
            grid_lines_width=grid_lines_width,
            grid_lines_color=grid_lines_color,
        )

    def plot_pairs(
        self,
        pairs: List[Dict],
        title: str,
        plot_size: int = 3,
        grid_lines_width: float = 0.3,
        grid_lines_color: str = "w",
    ) -> None:
        """Plot ARC pairs.

        :param pairs: list of dictionaries representing input/output pairs, as read from a JSON
        :param title: title to be added in plot
        :param plot_size: dimension of the plot
        :param grid_lines_width: floating-point width of lines separating pixels in the plot
        :param grid_lines_color: color to be used to visually separate grid cells.
            Matplotlib needs to recognise this as a valid color string, so for example
            the string "w" can be passed for white, and the string "k" can be passed for black.
        """
        num_pairs = len(pairs)
        assert num_pairs > 0

        if num_pairs == 1:
            self._plot_arc_one_pair(
                pairs, title, plot_size, grid_lines_width, grid_lines_color
            )
        else:
            self._plot_arc_more_than_one_pair(
                pairs, title, plot_size, grid_lines_width, grid_lines_color
            )

    @staticmethod
    def _get_arc_color(arc_color: int) -> Tuple[int, int, int]:
        "Transfom an ARC integer value into a uint8 RGB color."
        assert arc_color in set(range(11))

        RGB_COLORS = {
            0: (0, 0, 0),  # Black (#000000)
            1: (0, 116, 217),  # Blue (#0074D9)
            2: (255, 65, 54),  # Red (#FF4136)
            3: (46, 204, 64),  # Green (#2ECC40)
            4: (255, 220, 0),  # Yellow (#FFDC00)
            5: (170, 170, 170),  # Grey (#AAAAAA)
            6: (240, 18, 190),  # Fuchsia (#F012BE)
            7: (255, 133, 27),  # Orange (#FF851B)
            8: (127, 219, 255),  # Teal (#7FDBFF)
            9: (135, 12, 37),  # Brown (#870C25)
            10: (250, 250, 250),  # White (#FFFFFF)
        }
        return RGB_COLORS[arc_color]

    def _get_arc_image(self, grid: List[List[int]]) -> np.ndarray:
        """Create a ARC image from a `grid` contained in the ARC challenge JSON files."""
        num_rows = len(grid)
        num_cols = len(grid[0])

        image = np.zeros(shape=(num_rows, num_cols, 3), dtype=np.uint8)
        for i in range(num_rows):
            for j in range(num_cols):
                image[i, j] = self._get_arc_color(grid[i][j])

        return image

    def _plot_arc_more_than_one_pair(
        self,
        pairs: List[Dict],
        title: str,
        plot_size: int = 3,
        grid_lines_width: float = 0.3,
        grid_lines_color: str = "w",
    ) -> None:
        """Internal function used to plot multiple grid pairs in a ARC task."""
        num_pairs = len(pairs)
        fig, ax = plt.subplots(
            2, num_pairs, figsize=(plot_size * num_pairs, plot_size * 2)
        )
        fig.suptitle(title, fontsize=7)

        for index, pair in enumerate(pairs):
            input_image = self._get_arc_image(pair["input"])
            ax[0, index].imshow(input_image)
            self._draw_grid_lines(
                ax[0, index], input_image, grid_lines_width, grid_lines_color
            )
            ax[0, index].axis("off")

            output_image = self._get_arc_image(pair["output"])
            ax[1, index].imshow(output_image)
            self._draw_grid_lines(
                ax[1, index], output_image, grid_lines_width, grid_lines_color
            )
            ax[1, index].axis("off")

        fig.tight_layout()

        if self.savefig:
            touch_dir(self.dirsave)
            fig.savefig(os.path.join(self.dirsave, self.taskname), dpi=150)
            plt.close(fig)

    def _plot_arc_one_pair(
        self,
        pairs: List[Dict],
        title: str,
        plot_size: int = 3,
        grid_lines_width: float = 0.3,
        grid_lines_color: str = "w",
    ) -> None:
        """Internal function used to plot one single grid pair in a ARC task."""
        num_pairs = len(pairs)
        fig, ax = plt.subplots(
            2, num_pairs, figsize=(plot_size * num_pairs, plot_size * 2)
        )
        fig.suptitle(title, fontsize=14)

        pair = pairs[0]
        input_image = self._get_arc_image(pair["input"])
        ax[0].imshow(input_image)
        self._draw_grid_lines(ax[0], input_image, grid_lines_width, grid_lines_color)
        ax[0].axis("off")

        output_image = self._get_arc_image(pair["output"])
        ax[1].imshow(output_image)
        self._draw_grid_lines(ax[1], output_image, grid_lines_width, grid_lines_color)
        ax[1].axis("off")

        fig.tight_layout()

        if self.savefig:
            touch_dir(self.savedir)
            fig.savefig(os.path.join(self.savedir, self.taskname), dpi=150)
            plt.close(fig)

    @staticmethod
    def _draw_grid_lines(
        ax: matplotlib.axes.Axes,
        image: np.ndarray,
        grid_lines_width: float = 0.3,
        grid_lines_color: str = "w",
    ) -> None:
        """Draw grid lines between pixels in a ARC image."""
        num_rows, num_cols, _ = image.shape

        for i in range(num_rows + 1):
            ax.plot(
                [-0.52, num_cols - 0.52],
                [i - 0.52, i - 0.52],
                color=grid_lines_color,
                lw=grid_lines_width,
                zorder=2,
            )

        for j in range(num_cols + 1):
            ax.plot(
                [j - 0.52, j - 0.52],
                [-0.52, num_rows - 0.52],
                color=grid_lines_color,
                lw=grid_lines_width,
                zorder=2,
            )
