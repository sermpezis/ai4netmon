# Table of Contents

- [Introduction](#introduction)
- [Tool 1: Bias in Internet Measurement Platforms](#chapter-1-bias-in-internet-measurement-platforms)
- [Tool 2: Extending](#chapter-2-radar-plot-visualization)

## Introduction <a name="introduction"></a>

There are 5 visualization tools for the AI4NetMon project, in Observable notebooks, that the public can use in order to get various insights of collected data in IMPs, as well as in custom sets of ASNs. 

## Tool 1: Bias in Internet Measurement Platforms <a name="chapter-1-bias-in-internet-measurement-platforms"></a>

Provide detailed information about the bias present in Internet measurement platforms and the importance of addressing it.
This notebook visualizes the bias in Internet measurement platforms, based on the analyses of the AI4NetMon project, using radar plots. For an introduction on the bias of Internet measurement platforms see the [RIPE Labs article](https://labs.ripe.net/author/pavlos_sermpezis/bias-in-internet-measurement-infrastructure/).

The following radar plot shows:
show the possible selections the user can make

- the bias; from 0 (low bias, center of the circle) to 1 (high bias, far from center)
- for each selected set of monitors (lines/areas of different colors)
- along different bias dimensions (each radius of the circle)

![Example Image](figures/radarplot-imp-custom.png)

Also, the user can upload any custom set of monitors to compare their bias, except from the two IMPs, and select which sets of monitors / bias dimensions to visualize.
The bar on the top of the image is where the user puts the ASNs. Regarding those selections, the radar plot updates dynamically.

![Example Image](figures/selections1.png)

### Bias in RIPE RIS per Route Collector

In more detail, apart from the bias in IMPs, there is another visualization to show bias especially in RIPE RIS platform, per Route Collector. The user can use the selections in the screenshot below, to pick which Route Collectors appear in the radar plot.

![Example Image](figures/rrc-selections.png)

Again, the user can select which dimensions to appear on the radar plot. The corresponding radar plot is shown below:

![Example Image](figures/rrc-radarplpt.png)

As we can see, in the screenshot three dimensions are not selected, #IXPs, Scope and Personal ASN, so they are not visible in the radar plot either.

### Bias vs. volume of information

Some more interesting visualizations, using radar plots and line charts, concern the bias in subsets of peers of the RIPE RIS and RouteViews projects, where the subsets are selected by the Most Valuable Points (MVP) methodology of the [MVP paper](https://dl.acm.org/doi/abs/10.1145/3517745.3563031) and their online API.
Given a threshold of volume of information (e.g., RIBs of 5GBs) of number of vantage points (e.g., 100 peers), the MVP selects a subset of peers of RIPE RIS and RouteViews that are "most dissimilar" and bring "the more useful information with fewer data".

So, in the following visualization we compare the bias of (a) the entire set of RIPE RIS and RouteViews vantage points / peers with (b) their subset selected by MVP. The size of the subset can be selected; default value is 50 vantage points.

In the first plot, we see the bar to be used, in order to choose the number of vantage points returned by MVP.

![Example Image](figures/barformvp.png)

Then, the corresponding radar plot follows

![Example Image](figures/mvpradarplot.png)

Furthermore, the following figure, presents again the bias values per group of dimensions. For example the "Location" line corresponds to the average bias across the three dimensions "RIR region", "Country" and "Continent". The black line corresponds to the average bias (among all dimensions) of the MVP sets of vantage points vs. the number of vantage points (x-axis).

![Example Image](figures/mvplines.png)





