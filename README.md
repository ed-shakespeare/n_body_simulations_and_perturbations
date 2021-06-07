# n_body_project
Code for n body orbits/choreographies with Max Ruffert


## To run code

`n_body_v2.py` currently contains the script that can be adjusted for ICs. The ICs can be changed at the top of the file, and the visualisation instructions changed towards the bottom. When this file is run, it uses functions from the other files in the repository to produce visualisations of the trajectories, based on the inital conditions, masses of bodies, etc., as well as plots to show certain quantities over time.

## Possible configurations

Currently, the code can produce plots of trajectories with the following variable features:
- N >= 2 bodies, with freely configurable initial position, initial velocity, and mass
- 2D or 3D plot
- Animated or still graph
- Can plot energy, eccentricity, angular momentum, centre of mass, as functions of time or in 3D (where appropriate) *
- Can plot in frame of centre of mass
- Integration done through Euler, Euler-Cromer, or Leapfrog

\* Most features are relatively robust, although these can be prone to error in calculation
