# Gold Price Prediction Using Linear Regression

This repository contains a Python project that predicts today's gold prices using the past two days' prices. The project uses Linear Regression to model the relationship between the previous days' prices and today's price, utilizing **numpy**, **pandas**, **matplotlib**, and **scikit-learn** libraries. 

The project includes visualization of the dataset and the regression model in 3D, providing a clear understanding of the prediction process.

## Dataset
The dataset contains historical gold prices, specifically:
- `Price 2 Days Prior`
- `Price 1 Day Prior`
- `Price Today`

The target is to predict `Price Today` using the past two days' prices.
The data is taken from the kaggle Gold Prices dataset.
## Libraries Used
- **pandas** for data handling
- **numpy** for numerical operations
- **matplotlib** for data visualization
- **scikit-learn** for model building and evaluation

## Steps
1. **Data Preprocessing**: The dataset is loaded and the necessary features (prices from two prior days) are extracted.
2. **Model Building**: A Linear Regression model is trained using the `train_test_split` function to divide the data into training and testing sets.
3. **Evaluation**: The model's performance is evaluated using Mean Squared Error (MSE) and R² Score.
4. **Visualization**: 
   - A 3D scatter plot displays the training and testing data.
   - The Linear Regression plane is plotted to show how the model fits the data.

## Visualization
The project includes two graphs:
- **Gold Price Dataset**: A 3D scatter plot of the dataset with training and testing data.
- **Gold Price Prediction**: A 3D scatter plot along with the Linear Regression plane that represents the model.

## Usage
To run the code, simply execute the Python script. Ensure you have the following libraries installed:
```bash
pip install pandas numpy matplotlib scikit-learn
```
Replace the placeholder with the actual path to your dataset in the `read_csv` function.

## Example Output
- **Mean Squared Error**: Displays the error between the predicted and actual prices.
- **R² Score**: Represents the goodness of fit of the model.

## License
This project is open-source and available under the Apache License, Read the License file to read further.
