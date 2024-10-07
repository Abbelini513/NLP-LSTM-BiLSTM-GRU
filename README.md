# Text Classification with RNNs: LSTM, GRU, Bidirectional-LSTM

This repository is focused on building and comparing different Recurrent Neural Network (RNN) models for text classification. The task is to train models such as LSTM, GRU, and Bidirectional LSTM, and compare their performance using a custom evaluation metric. Additionally, heuristic approaches and classical machine learning methods are compared with RNNs.

## Task Overview

The goal of this project is to:

1. Train several recurrent neural network models (e.g., LSTM, GRU, Bidirectional-LSTM).
2. Evaluate the models using the custom metric proposed in Part 1.
3. Compare the results of RNNs with heuristic approaches and classical machine learning models.

### Key Points:
- Experiment with various hyperparameters and layers of RNNs to optimize performance.
- Validate that the model does not overfit by using a validation dataset.
- Present a conclusion on which approach worked best (RNN, heuristics, or classical ML).

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Abbelini/NLP-LSTM-BiLSTM-GRU.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Activate your virtual environment if necessary:

    ```bash
    source venv/bin/activate  # For Linux/macOS
    .\\venv\\Scripts\\activate  # For Windows
    ```

## Dataset

The dataset includes:

- `train.csv`: The training data for model building and validation.
- `test.csv`: The test data for final evaluation.
- `sample_submission.csv`: A sample submission file for generating predictions.

These datasets are used to train and evaluate the models, as well as test their generalization ability.

## Models

The following models are implemented:

1. **LSTM** (Long Short-Term Memory): A popular RNN variant used to capture long-range dependencies in sequential data.
2. **GRU** (Gated Recurrent Unit): A simpler variant of LSTM that often performs similarly with fewer parameters.
3. **Bidirectional-LSTM**: A model that processes input sequences in both forward and backward directions, potentially capturing more context.

### Hyperparameter Tuning
- The models can be tuned for various parameters such as:
  - Number of hidden layers and units.
  - Dropout rate.
  - Batch size.
  - Learning rate.
  - Sequence length (input size).

## Evaluation

A custom evaluation metric was designed in Part 1 of this project. The metric is used to evaluate the model's performance on the validation and test datasets. Each model's performance is compared across this metric.

### Heuristic and Classical ML Approaches
In addition to RNNs, heuristic methods and classical machine learning models (e.g., Naive Bayes, SVM) are also implemented for comparison. The results are benchmarked to determine which approach performs better.

### Validation
A validation set is used to monitor overfitting and assess generalization. Hyperparameters are adjusted based on validation results to avoid overfitting.

## Results

The results are compared for the following:

1. **RNN Models (LSTM, GRU, Bidirectional-LSTM)**: Evaluated on the custom metric and compared for accuracy, speed, and overfitting.
2. **Heuristic Methods**: Simple methods for baseline comparison.
3. **Classical Machine Learning Models**: Benchmark models to assess the added value of deep learning techniques.

### Conclusion:
After comparing all models, we draw a conclusion on which model or approach performed best and under what circumstances. We also provide insights on why certain models might have worked better than others.

## How to Run

1. Open the Jupyter Notebook:

    ```bash
    jupyter notebook PyTor4_NLP2.ipynb
    ```

2. Run the cells to:
    - Preprocess the data.
    - Train the models (LSTM, GRU, Bidirectional-LSTM).
    - Evaluate the models.
    - Compare the results with heuristics and classical ML methods.

3. Adjust hyperparameters as needed to experiment with different configurations.

## Contributing

If you'd like to contribute, feel free to fork this repository, submit issues, and create pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.
MIT License

Copyright (c) 2024 Alina Bondareva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

