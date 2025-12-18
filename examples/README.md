Please install a Python distribution, such as Miniconda (https://docs.anaconda.com/miniconda/). 

In command line:

1. Create a Python environment using `conda create -n transversity jupyterlab matplotlib`.

2. Execute `conda activate transversity` to activate the environment.

3. Install tParton via `pip install tparton`, or manually import the relevant package from the tParton/src/tparton folder.

Suggested run order is as follows:

1. "Generate Mathematica initial distributions.ipynb" is the Jupyter notebook which implements the GS-A type distribution Hirai used in their paper, and against which we verify our code. This file results in the input_diff.dat and input_upd.dat files.

2. "Evolve-Final.nb" is a Mathematica notebook containing the numerical validation of moments as well as the Mathematica implementation of Hirai's method. Results are saved to "mathematica.dat" and "mathematica_diff.dat".

3. "Running Hirai.ipynb" and "Running Vogelsang.ipynb" perform the evolutions using the analytical expressions for the strong coupling constant, saving their results to "hirai.npz" and "vogelsang.npz". The same files with "*-exact_alpha.ipynb" perform the evolutions using the numerically evolved strong coupling constant.

4. The "vogelsang_exact_alpha_variable_degree" folder varies the degree of approximation and input PDF granularity in Vogelsang evolution (Figure 4).

5. The "hirai_exact_alpha_nx_nt" folder varies the number of interpolation points and timesteps in Hirai evolution (Figure 5).

6. "Figures.ipynb" reproduces our figures.
