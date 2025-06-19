



## Getting Started

Follow the steps below to set up and run the code:

### Prerequisites

Ensure you have Python and the necessary libraries installed, including:
- torch
- numpy
- pandas
- matplotlib
- torchmetrics
- scikit-learn
- mediapipe

### Dataset
```
Dataset/
  User1/
    Session1
    Session2
    Session3
  User2/
  ...
  User21/
```
### Usage

#### Step 1: Preprocess the Data

To preprocess the data for data synchronization, run:

```bash
python preprocessing_data.py
python processing_ours.py
```

#### Step 2: Train and Evaluate the Model

To train and evaluate the model, execute:

```bash
python main_ours.py
```

#### Step 4: (Optional) Baseline Model

For the baseline model, move the code in the `Baseline/` folder to the parent folder and use it in the same way. Run the training and evaluation script:

```bash
python main_XXXX.py
```

### Project Structure

- `Baseline/`: Folder containing the code for the baseline model.
- `Dataset/`: Directory for organizing datasets.
- `Checkpoints/`: Directory for storing model-related files.
- `dataset.py`: File for handling the dataset.
- `joint_ops.py`: File for joint operations.
- `joint_ops_torch.py`: Torch implementation of joint operations.
- `main_ours.py`: Main script for training and evaluating the model.
- `models.py`: File containing model definitions.
- `preprocessing_data.py`: Script for preprocessing the dataset for data synchronization.
- `processing_ours.py`: Script for data processing and generating the dataset.
- `utils.py`: Utility functions.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
