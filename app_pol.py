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

st.set_page_config(page_title="Data Cleaning App", layout="wide")
st.title("🧹 Data Cleaning & Analysis App")

# Session state for data and cleaning history
def get_data():
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    return st.session_state['data']

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

# File upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key="file_uploader")

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    set_data(df)
    get_history().clear()
    add_history(df)
    st.success(f"Loaded {uploaded_file.name} with shape {df.shape}")

# Refresh df to ensure latest session state
df = get_data()

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {list(df.columns)}")
    st.write(df.dtypes)
    st.write("Current columns:", list(df.columns))

    # Cleaning options
    st.sidebar.header("2. Data Cleaning")
    with st.sidebar.expander("Drop Null Values"):
        with st.form("nulls_form"):
            drop_nulls = st.checkbox("Drop rows with any nulls", key="drop_nulls_checkbox")
            drop_nulls_cols = st.checkbox("Drop columns with any nulls", key="drop_nulls_cols_checkbox")
            fill_nulls = st.checkbox("Fill nulls", key="fill_nulls_checkbox")
            fill_value = st.text_input("Fill value (leave blank for mean/ffill)", key="fill_value_input")
            submitted = st.form_submit_button("Apply Null Handling", key="nulls_submit")
            if submitted:
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

    with st.sidebar.expander("Drop Duplicates"):
        with st.form("dupes_form"):
            drop_dupes = st.checkbox("Drop duplicate rows", key="drop_dupes_checkbox")
            subset_cols = st.multiselect("Subset columns for duplicate check", options=list(df.columns), key="dupe_subset_multiselect")
            submitted = st.form_submit_button("Apply Duplicate Handling", key="dupe_submit")
            if submitted:
                add_history(df)
                if drop_dupes:
                    if subset_cols:
                        df = df.drop_duplicates(subset=subset_cols)
                    else:
                        df = df.drop_duplicates()
                set_data(df)
                st.experimental_rerun()

    with st.sidebar.expander("Rename Columns"):
        with st.form("rename_form"):
            col_to_rename = st.selectbox("Column to rename", options=list(df.columns), key="rename_col_select")
            new_col_name = st.text_input("New column name", key="new_col_name_input")
            submitted = st.form_submit_button("Rename Column", key="rename_submit")
            if submitted and new_col_name:
                add_history(df)
                df = df.rename(columns={col_to_rename: new_col_name})
                set_data(df)
                st.experimental_rerun()

    with st.sidebar.expander("Change Data Types"):
        with st.form("type_form"):
            col_to_convert = st.selectbox("Column to convert", options=list(df.columns), key="type_col_select")
            dtype = st.selectbox("New type", options=["int", "float", "str", "datetime"], key="type_dtype_select")
            submitted = st.form_submit_button("Convert Type", key="type_submit")
            if submitted:
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

    with st.sidebar.expander("Filter Rows"):
        with st.form("filter_form"):
            col_to_filter = st.selectbox("Column to filter", options=list(df.columns), key="filter_col_select")
            unique_vals = df[col_to_filter].unique()
            filter_val = st.selectbox("Value to keep", options=unique_vals, key="filter_val_select")
            submitted = st.form_submit_button("Apply Filter", key="filter_submit")
            if submitted:
                add_history(df)
                df = df[df[col_to_filter] == filter_val]
                set_data(df)
                st.experimental_rerun()

    with st.sidebar.expander("Sort Data"):
        with st.form("sort_form"):
            sort_col = st.selectbox("Column to sort", options=list(df.columns), key="sort_col_select")
            ascending = st.checkbox("Ascending", value=True, key="sort_asc_checkbox")
            submitted = st.form_submit_button("Sort Data", key="sort_submit")
            if submitted:
                add_history(df)
                df = df.sort_values(by=sort_col, ascending=ascending)
                set_data(df)
                st.experimental_rerun()

    with st.sidebar.expander("Remove/Keep Columns"):
        with st.form("remove_form"):
            cols_to_remove = st.multiselect("Columns to remove", options=list(df.columns), key="remove_cols_multiselect")
            submitted_remove = st.form_submit_button("Remove Columns", key="remove_submit")
            if submitted_remove and cols_to_remove:
                add_history(df)
                df = df.drop(columns=cols_to_remove)
                set_data(df)
                st.experimental_rerun()
        with st.form("keep_form"):
            cols_to_keep = st.multiselect("Columns to keep", options=list(df.columns), key="keep_cols_multiselect")
            submitted_keep = st.form_submit_button("Keep Only Selected Columns", key="keep_submit")
            if submitted_keep and cols_to_keep:
                add_history(df)
                df = df[cols_to_keep]
                set_data(df)
                st.experimental_rerun()

    with st.sidebar.expander("Reset Index"):
        if st.button("Reset Index", key="reset_index"):
            add_history(df)
            df = df.reset_index(drop=True)
            set_data(df)
            st.experimental_rerun()

    st.sidebar.header("3. Undo")
    if st.sidebar.button("Undo Last Step", key="undo_btn"):
        undo()
        st.experimental_rerun()

    # Refresh df to ensure latest session state
    df = get_data()

    # The rest of the code below remains unchanged and uses the fresh df
    # ... [EDA, visualizations, profiling, ML, download sections] ...
