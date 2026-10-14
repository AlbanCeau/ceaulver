## ceaulver, a minimal collision solver for the GPU

This is a collision solver I developed for my own visual work, running on GPU. It's based on penalty methods and an energy conservative contact model, runs on Apple Silicon and also targets NVIDIA GPUs (untested), built using Taichi.

The solver handles box-shaped solids and is practical for real time at up to 2,000 active solids on an Apple Silicon M4 Pro.

I bundle a few example pieces with it, that can be modified to suit your own needs. All located in examples/, with pile.py, wall.py, and scree.py.

⟨prerequisites: uv is the only one; Python 3.12 is fetched by uv itself. Install command below⟩

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To run the examples: 

```
git clone https://github.com/AlbanCeau/ceaulver
cd ceaulver
uv run examples/wall.py
```

First run can be a bit slow as it requires uv to download some dependencies and compile the Taichi kernels.

Tested on macOS and Apple Silicon only. Linux needs the X11 library (libX11), Windows unknown

Examples have controls printed in the terminal.

The solver itself features a sleep logic (solids get their physics skipped below a threshold linear and angular speed), static and Coulomb friction between solids and between solids and their support.

Examples below are rendered with OpenGL, but the solver isn't renderer dependent. Solids are described in terms of their geometric center, half-extents in x, y and z and a rotation matrix, accessed through Solver.get_instance_data().

You can use examples/minimal.py to build your own pieces, or directly extract the solids' parameters from the Solver class.



https://github.com/user-attachments/assets/ee09cd72-4594-4f98-884c-da724226ffac

5,000 active solids at 60 FPS on an Apple Silicon M4 Pro. The cliff is eroded using a Perlin noise to decide the order in which solids exit sleep.



https://github.com/user-attachments/assets/0593b560-5e2a-4137-97f5-877a91677e0e

A heap built one solid at a time, the impact of the sleep logic can be observed by forcing all solids awake during the run, as can the presence or absence of static friction.



https://github.com/user-attachments/assets/3db92012-eb28-4cad-9ff4-2063268b5c9b

An approximation of a scree (it needs time to build up to the avalanche shown in this clip), as with the other clips, the sleep logic can be observed at work by cycling between color modes using 'C', the top left panel allows for live modifications of the sim's parameters. 

⟨contact model reference, lead-in sentence⟩

- Feng, Y. T. (2021). *An energy-conserving contact theory for discrete element modelling of arbitrarily shaped particles: Basic framework and general contact model.* Comput. Methods Appl. Mech. Engrg. **373**, 113454. https://doi.org/10.1016/j.cma.2020.113454
- Feng, Y. T. (2021). *An energy-conserving contact theory for discrete element modelling of arbitrarily shaped particles: Contact volume based model and computational issues.* Comput. Methods Appl. Mech. Engrg. **373**, 113493. https://doi.org/10.1016/j.cma.2020.113493

⟨licence: one line, MIT, see LICENSE⟩
