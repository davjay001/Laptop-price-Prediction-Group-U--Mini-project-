# Laptop Price Prediction Project

This project predicts laptop prices using machine learning and provides a user-friendly interface via Streamlit.

## Project Structure

```
dataset/
    cleaned_laptop_price.csv
    laptop_price.csv
encoders/
    label_encoders.pkl
    laptop_price_model.pkl
    model_columns.pkl
notebook/
    mini_project.ipynb
    model_building.ipynb
streamlit/
    app.py
```

## Workflow

1. **Data Preparation**
    - Raw data is stored in `dataset/laptop_price.csv` and cleaned in `dataset/cleaned_laptop_price.csv`.
    - Data cleaning and feature engineering are performed in Jupyter notebooks.

2. **Model Building**
    - Categorical features are encoded using `OneHotEncoder`.
    - Numerical features are scaled using `StandardScaler`.
    - The model is trained using `LinearRegression`.
    - Model and encoders are saved as `.pkl` files in the `encoders/` directory.

3. **Streamlit App**
    - The app (`streamlit/app.py`) loads the trained model and encoders.
    - Users input laptop specifications via dropdowns and number inputs.
    - Only valid categories from the encoder are selectable, preventing prediction errors.
    - The app encodes and scales user input, then predicts and displays the laptop price.
    - Negative predictions are prevented and shown as €0.00 if they occur.

## How to Run

1. Install dependencies:
    ```bash
    pip install streamlit pandas scikit-learn joblib
    ```
2. Start the Streamlit app:
    ```bash
    cd streamlit
    streamlit run app.py
    ```

## Notebooks
- `notebook/model_building.ipynb`: Data cleaning, feature engineering, model training, and evaluation.
- `notebook/mini_project.ipynb`: Additional analysis and project documentation.

## Notes
- Ensure all `.pkl` files are present in the `encoders/` directory.
- Only select options provided by the app for categorical features to avoid errors.
- For best results, use realistic values within the range of the training data.

## Author
Group U

---
For questions or issues, please contact the project maintainers.
