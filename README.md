# ResoNet

ResoNet is an innovative neural network-based project designed to solve reasoning problems using a novel architecture called Y_NET. The Y_NET architecture is shaped like the letter "Y" and is inspired by denoising techniques. However, instead of merely denoising, it improves images by reasoning.

## Project Structure

```
ResoNet/
│
├── Y_NETS/
│   ├── resonet_v1.py
│   └── resonet_v2.py
│
├── dataset/
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_test_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   └── arc-agi_evaluation_challenges.json
│
├── main.py
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ResoNet.git
    cd ResoNet
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the `main.py` script. This script loads the training data, initializes the model, and trains it using the specified parameters.

```python
python main.py
```

### Evaluating the Model

The `main.py` script also includes functionality to evaluate the model on test data. The evaluation results are saved as images in the specified output directory.

### Visualizing Results

The `Y_NET` class includes a `cplot` method to visualize the grid, pairs of grids, or a sequence of images. This method can be used to generate and save plots of the model's predictions.

## Y_NET Architecture

The Y_NET architecture is designed to handle grid-based reasoning problems. It consists of an encoder-decoder structure that processes the input grid and generates an improved output grid by reasoning through the data. The architecture is shaped like the letter "Y", symbolizing the branching and merging of information.

### Key Features

- **Encoder**: Encodes the input grid into a latent representation.
- **Decoder**: Decodes the latent representation back into an improved grid.
- **Reasoning**: The model improves the grid by reasoning through the data, rather than just denoising it.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [ARC Dataset](https://github.com/fchollet/ARC)
```
