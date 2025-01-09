# Differentiable-multi-output-hydrological-model
This project provides a hybrid hydrological model that can output intermediate hydrological variables to provide a basis for studying the similarity of hydrological behavior in catchments. It provides tools for data processing, model training, testing and analysis of hydrological outputs with physical constraints.
## Project Structure
### Core Scripts
1.JOH_Regional_dPL_output_physics.py  

  Purpose: Evaluates the hydrological model's output, considering physical constraints.  
  
  oKey Functionality: Produces detailed model outputs for analysis.  
  
2.JOH_Regional_dPL_test_all_change_met+attrs.py  
  Purpose: The model was tested using 5 meteorological and 27 attribute data.  
  Key Functionality: Validates model performance across diverse catchments.  
3.JOH_Regional_dPL_train_all_change_met+attrs.py  
  Purpose: Trains the model with meteorological and basin attribute data.  
  Key Functionality: Optimizes model parameters using provided datasets.  
4.dPL_class.py  
  Purpose: Defines the architecture and components of the differentiable hydrological model.  
  Key Functionality: Encapsulates model layers, forward propagation, and parameter tuning logic.  
5.dataprocess.py  
  Purpose: Processes raw hydrological and meteorological datasets for training and testing.  
  Key Functionality: Ensures data normalization, feature extraction, and input formatting.  
6.loss.py  
  Purpose: Implements custom loss functions for model optimization.  
  Key Functionality: Supports physics-informed loss terms to enhance prediction accuracy.  
## Installation and Setup
### Prerequisites
  Python: Version 3.8 or later.  
  Recommended packages: numpy, pandas, torch, matplotlib, and scipy.  
### Setup Steps
1.Clone the repository:
  git clone https://github.com/Yq-H47/Differentiable-multi-output-hydrological-model
  cd Differentiable-multi-output-hydrological-model
2.Install dependencies:
  pip install -r environments.yml
  (If environments.yml is not provided, manually install the required libraries.)
3. Prepare datasets:
  Place your meteorological and hydrological data files in the data/ directory.
  Ensure data format matches the specifications in dataprocess.py.
## Usage Guide
### Model Training
To train the model with your dataset:
  python JOH_Regional_dPL_train_all_change_met+attrs.py
  Key Parameters:
    --epochs: Number of training iterations.
    --learning_rate: Learning rate for optimizer.
### Model Testing
To test the trained model:
  python JOH_Regional_dPL_test_all_change_met+attrs.py
  Key Outputs:
    Performance metrics (e.g., NSE, RMSE).
    Visualization of predictions versus ground truth.
### Evaluate Physical Constraints
To analyze model outputs with physical constraints:
  python JOH_Regional_dPL_output_physics.py
  Key Outputs:
    Predicted runoff and other intermediate hydrological variables.
---
## Example Workflow
1.Preprocess the data using dataprocess.py.
2.from dataprocess import preprocess_data
  preprocess_data("path_to_raw_data")
3.Train the model:
  python JOH_Regional_dPL_train_all_change_met+attrs.py --epochs 50 --learning_rate 0.001
4.Test the model:
  python JOH_Regional_dPL_test_all_change_met+attrs.py
5.Analyze results:
  python JOH_Regional_dPL_output_physics.py
---
## Outputs
**Metrics**: Evaluation metrics such as Nash-Sutcliffe Efficiency (NSE).
**Visualizations**: Predicted versus observed hydrological outputs.
**Logs**: Detailed logs of training and testing performance.
---
## Notes and Limitations
  Ensure datasets are preprocessed correctly using the provided scripts.
  Modify hyperparameters in training scripts for optimal performance on your dataset.
  For large datasets, training may require substantial computational resources.
---
## Contact
For further assistance or implementation details, please contact the author at: lh_mygis@163.com or 17563403791@163.com
