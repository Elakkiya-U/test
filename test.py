import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Digital Marketing Campaign Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Digital Marketing Campaign Analysis")
st.markdown("This application analyzes digital marketing campaign data and builds predictive models for conversion prediction.")

# Sidebar for file upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your digital marketing campaign dataset (CSV)",
    type=['csv'],
    help="Please upload a CSV file containing your digital marketing campaign data"
)

# Main application logic
if uploaded_file is not None:
    try:
        # Load the dataset
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        
        # --- Exploratory Data Analysis (EDA) ---
        st.header("🔍 Exploratory Data Analysis")
        
        # Dataset overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col3:
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Display basic info
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe())
        
        # Categorical variable distributions
        st.subheader("Categorical Variable Distributions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Gender Distribution**")
            st.dataframe(df["Gender"].value_counts())
        
        with col2:
            st.write("**Campaign Channel Distribution**")
            st.dataframe(df["CampaignChannel"].value_counts())
        
        with col3:
            st.write("**Campaign Type Distribution**")
            st.dataframe(df["CampaignType"].value_counts())
        
        # Check skewness of numerical features
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        skewness = df[numerical_cols].apply(lambda x: skew(x))
        
        st.subheader("Skewness of Numerical Features")
        st.dataframe(skewness.to_frame('Skewness'))
        
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        
        # --- Data Preprocessing ---
        st.header("⚙️ Data Preprocessing")
        
        with st.expander("Data Preprocessing Steps", expanded=True):
            # Drop irrelevant columns
            df_processed = df.drop(['CustomerID', 'AdvertisingPlatform', 'AdvertisingTool'], axis=1)
            st.write("✅ Dropped irrelevant columns: CustomerID, AdvertisingPlatform, AdvertisingTool")
            
            # Define features and target
            X = df_processed.drop('Conversion', axis=1)
            y = df_processed['Conversion']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            st.write("✅ Split data into training (80%) and testing (20%) sets")
            
            # --- Feature Engineering ---
            # Encode categorical variables
            categorical_cols = ['Gender', 'CampaignChannel', 'CampaignType']
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]), 
                                         columns=encoder.get_feature_names_out(categorical_cols), 
                                         index=X_train.index)
            X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]), 
                                        columns=encoder.get_feature_names_out(categorical_cols), 
                                        index=X_test.index)
            
            # Drop original categorical columns and concatenate encoded ones
            X_train = X_train.drop(categorical_cols, axis=1)
            X_test = X_test.drop(categorical_cols, axis=1)
            X_train = pd.concat([X_train, X_train_encoded], axis=1)
            X_test = pd.concat([X_test, X_test_encoded], axis=1)
            st.write("✅ Encoded categorical variables using One-Hot Encoding")
            
            # Handle skewness for highly skewed features
            skewed_cols = skewness[abs(skewness) > 1].index
            for col in skewed_cols:
                if col in X_train.columns and col != 'Conversion':
                    X_train[col + '_skewed'] = np.log1p(X_train[col])
                    X_test[col + '_skewed'] = np.log1p(X_test[col])
            
            # Drop original skewed columns if transformed
            for col in skewed_cols:
                if col in X_train.columns and col != 'Conversion':
                    X_train = X_train.drop(col, axis=1)
                    X_test = X_test.drop(col, axis=1)
            
            if len(skewed_cols) > 0:
                st.write(f"✅ Applied log transformation to {len(skewed_cols)} skewed features")
            
            # Apply SMOTE to handle class imbalance
            st.write("**Class Distribution Before SMOTE:**")
            before_smote = Counter(y_train)
            st.write(f"Class 0: {before_smote[0]}, Class 1: {before_smote[1]}")
            
            smote = SMOTE(random_state=42)
            try:
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                after_smote = Counter(y_train_smote)
                st.write("**Class Distribution After SMOTE:**")
                st.write(f"Class 0: {after_smote[0]}, Class 1: {after_smote[1]}")
                st.write("✅ Applied SMOTE for class balancing")
            except ValueError as e:
                st.error(f"Error during SMOTE: {e}")
                st.stop()
            
            # Scale features
            scaler = StandardScaler()
            X_train_smote = scaler.fit_transform(X_train_smote)
            X_test = scaler.transform(X_test)
            st.write("✅ Standardized features using StandardScaler")
        
        # --- Model Training ---
        st.header("🤖 Model Training & Evaluation")
        
        models = {
            "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_leaf_nodes=30, random_state=42),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42)
        }
        
        # Create tabs for each model
        tab1, tab2, tab3 = st.tabs(["Logistic Regression", "Random Forest", "Decision Tree"])
        
        tabs = [tab1, tab2, tab3]
        model_names = list(models.keys())
        
        for i, (name, model) in enumerate(models.items()):
            with tabs[i]:
                st.subheader(f"{name} Results")
                
                try:
                    # Train model
                    model.fit(X_train_smote, y_train_smote)
                    y_pred = model.predict(X_test)
                    
                    # Evaluation metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                        st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
                    
                    with col2:
                        st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
                        st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f"Confusion Matrix - {name}")
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1')
                    st.subheader("Cross-Validation Results")
                    st.write(f"**F1 Scores:** {cv_scores}")
                    st.write(f"**Mean CV F1 Score:** {cv_scores.mean():.4f}")
                    
                except Exception as e:
                    st.error(f"Error training/evaluating {name}: {e}")
        
        # --- Error Checking ---
        st.header("⚠️ Error Checking")
        
        error_checks = []
        
        if df.isnull().sum().sum() > 0:
            error_checks.append("⚠️ Missing values detected in the dataset")
        else:
            error_checks.append("✅ No missing values in the dataset")
        
        if df.duplicated().sum() > 0:
            error_checks.append("⚠️ Duplicate rows detected in the dataset")
        else:
            error_checks.append("✅ No duplicate rows in the dataset")
        
        if X_train_smote.shape[1] != X_test.shape[1]:
            error_checks.append("❌ Feature mismatch between training and test sets")
        else:
            error_checks.append("✅ No feature mismatch between training and test sets")
        
        if Counter(y_test)[0] == 0 or Counter(y_test)[1] == 0:
            error_checks.append("❌ Test set contains only one class")
        else:
            error_checks.append("✅ Test set contains both classes")
        
        error_checks.append("✅ Script executed successfully")
        
        for check in error_checks:
            st.write(check)
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.info("Please make sure your CSV file contains the required columns.")

else:
    st.info("👆 Please upload a CSV file to begin the analysis.")
    st.markdown("""
    ### Expected Dataset Format:
    Your CSV file should contain the following columns:
    - **CustomerID**: Unique identifier for customers
    - **Gender**: Customer gender
    - **CampaignChannel**: Marketing channel used
    - **CampaignType**: Type of campaign
    - **AdvertisingPlatform**: Platform used for advertising
    - **AdvertisingTool**: Tool used for advertising
    - **Conversion**: Target variable (0 or 1)
    - Other numerical features related to the campaign
    
    ### How to Use:
    1. Upload your digital marketing campaign dataset using the file uploader in the sidebar
    2. The app will automatically perform exploratory data analysis
    3. View data preprocessing steps and feature engineering
    4. Compare model performance across different algorithms
    5. Check for potential data quality issues
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")