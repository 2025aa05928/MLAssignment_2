# Sample Test Data - Information

## File: `test_data_sample.csv`

This sample CSV file is provided for testing the dataset upload feature in the Streamlit app.

### Dataset Details:

- **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Samples:** 100 rows
- **Features:** 30 numeric features
- **Target Column:** `target` (0 = Malignant, 1 = Benign)
- **File Size:** ~21 KB
- **Format:** CSV (Comma-Separated Values)

### Features Included:

All 30 features from the Breast Cancer dataset:
- mean radius, mean texture, mean perimeter, mean area
- mean smoothness, mean compactness, mean concavity
- mean concave points, mean symmetry, mean fractal dimension
- radius error, texture error, perimeter error, area error
- smoothness error, compactness error, concavity error
- concave points error, symmetry error, fractal dimension error
- worst radius, worst texture, worst perimeter, worst area
- worst smoothness, worst compactness, worst concavity
- worst concave points, worst symmetry, worst fractal dimension

### How to Use:

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload the sample CSV:**
   - In the sidebar, check ☑️ "Upload custom CSV"
   - Click "Browse files" or drag and drop `test_data_sample.csv`
   - Select `target` as the target column from the dropdown

3. **Configure settings:**
   - Adjust test size (default: 0.2)
   - Enable/disable feature scaling (recommended: enabled)
   - Set random state for reproducibility

4. **Train models:**
   - Select a specific model or choose "All Models"
   - Click 🚀 "Train Models"
   - Wait for training to complete

5. **View results:**
   - Evaluation metrics table
   - ROC curves comparison
   - Confusion matrices
   - Performance comparison charts

### Note:

This is a **test sample** extracted from the main Breast Cancer dataset. It contains only 100 samples (compared to the full 569 samples) to allow for quick testing of the upload functionality.

For the actual assignment submission, the app uses the complete Breast Cancer dataset (569 samples) loaded directly via `sklearn.datasets.load_breast_cancer()`.

### Assignment Requirement:

As per the assignment instructions:
> "Dataset upload option (CSV) [As streamlit free tier has limited capacity, upload only test data]"

This sample file serves as the test data for demonstrating the CSV upload feature while keeping within Streamlit Community Cloud's free tier limitations.

---

**Created:** February 15, 2026  
**Purpose:** Assignment 2 - ML Classification Models  
**Course:** M.Tech (AIML/DSE) Machine Learning - BITS Pilani
