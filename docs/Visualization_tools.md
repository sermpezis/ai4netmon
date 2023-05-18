# Table of Contents

- [Introduction](#introduction)
- [Tool 1: Bias in Internet Measurement Platforms](#chapter-1-bias-in-internet-measurement-platforms)
- [Tool 2: Radar Plot Visualization](#chapter-2-radar-plot-visualization)
- [Chapter 3: Instructions](#chapter-3-instructions)
- [Chapter 4: How to Run](#chapter-4-how-to-run)
- [Chapter 5: Acknowledgments](#chapter-5-acknowledgments)
- [Chapter 6: License](#chapter-6-license)

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
The bar on the top of the image is where the user puts the ASNs. 

![Example Image](figures/selections1.png)



