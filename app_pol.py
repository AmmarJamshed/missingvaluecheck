import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import base64
import os

# Set page config
st.set_page_config(page_title="Data Cleaning App", layout="wide")
st.title("🧹 Data Cleaning & Analysis App")

# Helper functions for session state
def get_data():
    return st.session_state.get('data', None)

def set_data(df):
    st.session_state['data'] = df

def get_history():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    return st.session_state['history']

def add_history(df):
    get_history().append(df.copy())

def undo():
    history = get_history()
    if len(history) > 1:
        history.pop()
        set_data(history[-1])

# Sidebar - Upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Load uploaded file only once
if uploaded_file and 'uploaded' not in st.session_state:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    set_data(df)
    get_history().clear()
    add_history(df)
    st.session_state['uploaded'] = True
    st.success(f"Loaded {uploaded_file.name} with shape {df.shape}")

# Get latest df
df = get_data()

# If data is available
if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    st.write(df.dtypes)

    # Sidebar - Data Cleaning Options
    st.sidebar.header("2. Data Cleaning")
    
    # Drop Nulls
    with st.sidebar.expander("Drop Null Values"):
        drop_nulls = st.checkbox("Drop rows with any nulls")
        drop_nulls_cols = st.checkbox("Drop columns with any nulls")
        fill_nulls = st.checkbox("Fill nulls")
        fill_value = st.text_input("Fill value (leave blank for mean/ffill)")
        if st.button("Apply Null Handling"):
            add_history(df)
            if drop_nulls:
                df = df.dropna()
            if drop_nulls_cols:
                df = df.dropna(axis=1)
            if fill_nulls:
                if fill_value:
                    df = df.fillna(fill_value)
                else:
                    for col in df.select_dtypes(include=[np.number]).columns:
                        df[col] = df[col].fillna(df[col].mean())
                    for col in df.select_dtypes(include=[object]).columns:
                        df[col] = df[col].fillna(method='ffill')
            set_data(df)
            st.experimental_rerun()

    # Drop Duplicates
    with st.sidebar.expander("Drop Duplicates"):
        drop_dupes = st.checkbox("Drop duplicate rows")
        subset_cols = st.multiselect("Subset columns for duplicate check", options=list(df.columns))
        if st.button("Apply Duplicate Handling"):
            add_history(df)
            if drop_dupes:
                if subset_cols:
                    df = df.drop_duplicates(subset=subset_cols)
                else:
                    df = df.drop_duplicates()
            set_data(df)
            st.experimental_rerun()

    # Rename Columns
    with st.sidebar.expander("Rename Columns"):
        col_to_rename = st.selectbox("Column to rename", options=list(df.columns))
        new_col_name = st.text_input("New column name")
        if st.button("Rename Column"):
            if new_col_name:
                add_history(df)
                df = df.rename(columns={col_to_rename: new_col_name})
                set_data(df)
                st.experimental_rerun()

    # Change Data Types
    with st.sidebar.expander("Change Data Types"):
        col_to_convert = st.selectbox("Column to convert", options=list(df.columns))
        dtype = st.selectbox("New type", options=["int", "float", "str", "datetime"])
        if st.button("Convert Type"):
            add_history(df)
            try:
                if dtype == "int":
                    df[col_to_convert] = df[col_to_convert].astype(int)
                elif dtype == "float":
                    df[col_to_convert] = df[col_to_convert].astype(float)
                elif dtype == "str":
                    df[col_to_convert] = df[col_to_convert].astype(str)
                elif dtype == "datetime":
                    df[col_to_convert] = pd.to_datetime(df[col_to_convert])
                set_data(df)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Filter Rows
    with st.sidebar.expander("Filter Rows"):
        col_to_filter = st.selectbox("Column to filter", options=list(df.columns))
        unique_vals = df[col_to_filter].unique()
        filter_val = st.selectbox("Value to keep", options=unique_vals)
        if st.button("Apply Filter"):
            add_history(df)
            df = df[df[col_to_filter] == filter_val]
            set_data(df)
            st.experimental_rerun()

    # Sort Data
    with st.sidebar.expander("Sort Data"):
        sort_col = st.selectbox("Column to sort", options=list(df.columns))
        ascending = st.checkbox("Ascending", value=True)
        if st.button("Sort Data"):
            add_history(df)
            df = df.sort_values(by=sort_col, ascending=ascending)
            set_data(df)
            st.experimental_rerun()

    # Remove / Keep Columns
    with st.sidebar.expander("Remove/Keep Columns"):
        cols_to_remove = st.multiselect("Columns to remove", options=list(df.columns))
        if st.button("Remove Columns"):
            if cols_to_remove:
                add_history(df)
                df = df.drop(columns=cols_to_remove)
                set_data(df)
                st.experimental_rerun()
        cols_to_keep = st.multiselect("Columns to keep", options=list(df.columns))
        if st.button("Keep Only Selected Columns"):
            if cols_to_keep:
                add_history(df)
                df = df[cols_to_keep]
                set_data(df)
                st.experimental_rerun()

    # Reset Index
    if st.sidebar.button("Reset Index"):
        add_history(df)
        df = df.reset_index(drop=True)
        set_data(df)
        st.experimental_rerun()

    # Undo Button
    st.sidebar.header("Undo Changes")
    if st.sidebar.button("Undo Last Change"):
        undo()
        st.experimental_rerun()
