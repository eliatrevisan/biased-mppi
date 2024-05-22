# biased-mppi
This repository presents examples where MPPI's sampling distribution is informed with ancillary controllers.
Adding these ancillary controllers (or priors) makes MPPI more efficient and less prone to local minima.

You can find more information on the [paper's website](https://autonomousrobots.nl/paper_websites/biased-mppi).
This repository only contains examples, as biased-mppi has been implemented in our planner [mppi-torch](https://github.com/tud-airlab/mppi_torch).

## Structure

The project is structured as follows:

- `examples/`: Contains motion planning examples.
- `pyproject.toml` and `poetry.lock`: Configuration files for dependencies.

## Installation

To install the project, follow these steps:

```sh
# Clone the repository
git clone <repository-url>

# Navigate to the project directory
cd <project-directory>

# Install dependencies
poetry install
```

Access the virtual environment using
```bash
poetry shell
```

Requires poetry ^1.8.

## Usage

To run the point robot example:

```
cd examples/jackal
python run.py
```

## Contributing

Contributions are welcome. Please submit a pull request.

## Cite

If you find this code useful, please cite:
```bash
@ARTICLE{biased-mppi,
  author={Trevisan, Elia and Alonso-Mora, Javier},
  journal={IEEE Robotics and Automation Letters}, 
  title={Biased-MPPI: Informing Sampling-Based Model Predictive Control by Fusing Ancillary Controllers}, 
  year={2024},
  volume={9},
  number={6},
  pages={5871-5878},
  keywords={Costs;Planning;Monte Carlo methods;Mathematical models;Optimal control;Vehicle dynamics;Trajectory;Motion and path planning;optimization and optimal control;collision avoidance;sampling-based MPC;MPPI},
  doi={10.1109/LRA.2024.3397083}}
```
