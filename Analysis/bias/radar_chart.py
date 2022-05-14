# The code in this based on the code downloaded from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html (09 Jan 2022)

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta




def plot_radar_from_dataframe(df, colors=None, frame='polygon', cmap='tab10', rgrids=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], alpha=0.25, save_filename=None, show=False, fontsize='small', fontsize_features='small', varlabels=None, legend_loc=(0.9, .95)):
    '''
    Generates a radar plot from the given dataframe (df) with axes the rows of the df, surfaces the columns of the df
    and the values along each axis correspond to the values of the df. 
    It saves the plot if a "save_filename" is given and/or shows the plot if the "show" is set to True

    :param  df:             (pandas.Dataframe) with indexes the axes of the plot, columns the surfaces
    :param  colors:         (list) define a list of colors; if None (default), use the given cmap colors
    :param  cmap:           (string) the cmap of the colors to be used; if colors is not None, it is not taken into account
    :param  save_filename:  (string) the filename to save the generated plot
    :param  show:           (boolean) if True it shows the plot
    :param  varlabels:      (dict) dict with mapping of feature names in the df (str) to names appearing in the plot (str)

    the other parameters are set for formatting the plot
    '''

    case_data = df.to_numpy().transpose() # get the data in numpy format
    features = df.index
    labels = df.columns
    N = len(features)   # get the number of features (i.e., dimensions in the radar plot)
    M = len(labels)
    theta = radar_factory(N, frame=frame)   # get the radar plot 
    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95, bottom=0.05)#, right=0.85, left=0.15)

    if colors is None:
        cmap = get_cmap(cmap)
        if M>10:
            norm = Normalize(vmin=0, vmax=M-1)
        else:
            norm = Normalize(vmin=0, vmax=10)
        colors = [cmap(norm(i)) for i in range(M)]
    ax.set_rgrids(rgrids, fontsize=fontsize)

    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
    legend = ax.legend(labels, loc=legend_loc, labelspacing=0.1, fontsize=fontsize)

    for d, color in zip(case_data, colors):
        ax.fill(theta, d, facecolor=color, alpha=alpha)

    if varlabels is None:
        ax.set_varlabels(features,fontsize=fontsize_features)
    else:
        ax.set_varlabels([varlabels[i] for i in features],fontsize=fontsize_features)

    if save_filename is not None:
        plt.savefig(save_filename)
    if show:
        plt.show()
    plt.close()