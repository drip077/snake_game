# Snake Game AI

This repository contains the implementation of an AI that plays the Snake game. The AI is trained using reinforcement learning techniques.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone this repository:**
   ```sh
   git clone https://github.com/yourusername/snake_game_ai.git
   cd snake_game_ai
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the Snake game AI model, run:
   ```sh
   python train.py
   ```

### Running the Model

To run the trained model and watch it play the Snake game, run:
   ```sh
   python run_model.py
   ```

## Files

- `README.md`: This file, containing information about the project.
- `agent.py`: Contains the implementation of the agent that interacts with the game environment.
- `game.py`: Contains the implementation of the Snake game environment.
- `helper.py`: Contains helper functions used throughout the project.
- `model.py`: Contains the neural network model used by the agent.
- `run_model.py`: Script to run the trained model and watch it play the game.
- `train.py`: Script to train the model using reinforcement learning techniques.


