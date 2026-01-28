# Injective_representative_project
The code for my final year UON research project that aims to find an injective representative of a multi-dimensional input function defined over a high-dimensional degenerate input space.

There are three sets of files: PIC, Line-Slice and BCD.

The PIC code is run from main and it uses functions from Optimiation_classes, Sim_function and Lasy. The line_slice algorithm is contained within line_slice_main and the results are inverted with neural_inversion. There are scripts for single and multi objective Bayesian Contour Descent with a few helper functions stored in Objective_grid_search. GPU_BCD is the GPU compatible version of multi_objective_BCD.
