All code that is needed for running simulations is in highest level, with other versions\old code containing outdated versions of certain files

The files available are:

- Plot_all_trajectories.py
    - This can be used to simulate every type of solution seen in the paper.
- gridsearch_perturbations_fig8.ipynb
    - Produces plots after gridsearch of perturbations to fig8
    - Jupyter notebook, to make it easier to store outputs once the gridsearch has taken place and then modify graphs etc without rerunning all code
- gridsearch_perturbations_butterfly.ipynb
    - Similar to above, but for the butterfly solution
- single_perturbation_fig8.ipynb
    - Plots the trajectory of a single perturbations
    - Used for inspection of specific solutions, without needing to research entire grid
- single_perturbation_butterfly.ipynb
    - Similar to above, but for the butterfly solution

These all call functions from other files:

- ode_function.py
    - Contains the functions found in the system of ODEs
- test_constants.py
    - A range of functions for testing properties of the system that should remain constant, such as energy, angular momentum
- integration_methods.py:
    - The three different integration methods chosen for this investigation
