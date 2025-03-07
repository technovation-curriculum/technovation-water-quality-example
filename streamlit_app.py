import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from catboost import CatBoostClassifier

indicator_thresholds = {
    'Chl-a': {
        'type': 'above',
        'moderate': 10,
        'high_threshold': 25
    }, 
    'Cyanobacteria (total volume)': {
         'type': 'above',
         'moderate': 0.2,
        'high_threshold': 4       
    },
    'Cyanobacteria (biovolume equiv of potentially toxic)': {
        'type': 'above',
        'moderate': 0.2,
        'high_threshold': 4          
    }, 
    'E. coli': {
         'type': 'above',
        'moderate': 126,
        'high_threshold': 235   
    },
    'Ammoniacal Nitrogen': {
        'type': 'above',
        'moderate': 0.5,
        'high_threshold': 1.0   
    },       
    'pH': {
        'type': 'outside_range',
        'moderate': [5.5, 9.0],
        'high_threshold': [6.5, 8.5]
    },       
    'Secchi': {
        'type': 'below',
        'moderate': 1.2,
        'high_threshold': 0.5,
    },   
    'Total Nitrogen': {
        'type': 'above',
        'moderate': 0.5,
        'high_threshold': 1.0   
    },
    'Total Phosphorus': {
        'type': 'above',
        'moderate': 0.02,
        'high_threshold': 0.05
    }
}

@st.cache_data
def read_data():
    # read csv file and format date for processing
    df = pd.read_csv('lawa-lake-monitoring-data-2004-2023_statetrendtli-results_sep2024.csv')
    
    # Convert SampleDateTime to datetime
    df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'])
    
    # Extract date and month for analysis
    df['Date'] = df['SampleDateTime'].dt.date
    df['Month'] = df['SampleDateTime'].dt.month
    df['Year'] = df['SampleDateTime'].dt.year    
    return df

def determine_risk_level(row):
    """
    Determine risk level for a single value based on indicator conditions.
    
    Returns:
    --------
    int
        Risk level (0: Safe, 1: Moderate, 2: High)
    """
    # Handle potential None or empty condition
    indicator = row['Indicator']
    condition = indicator_thresholds.get(indicator)
    value = pd.to_numeric(row['Value (Agency)'], errors='coerce')
    # print(f"finding risk value for row, value is {value}")
    if not condition or value is None:
        return 0
    
    condition_type = condition.get("type", "above")
    moderate_threshold = condition.get("moderate")
    
    # Handle both 'high_threshold' and 'threshold' keys
    high_threshold = condition.get("high_threshold") or condition.get("threshold")
    
    # Default risk is safe
    risk_level = 0
    
    # Risk determination logic for different condition types
    if condition_type == "above":
        # Prioritize high threshold if present
        if high_threshold is not None and value > high_threshold:
            risk_level = 2  # High risk
            # st.write(f"High Risk: Value {value} > High Threshold {high_threshold}")
        elif moderate_threshold is not None and value > moderate_threshold:
            risk_level = 1  # Moderate risk
            # st.write(f"Moderate Risk: Value {value} > Moderate Threshold {moderate_threshold}")
    
    elif condition_type == "below":
        # Prioritize high threshold if present
        if high_threshold is not None and value < high_threshold:
            risk_level = 2  # High risk
            # st.write(f"High Risk: Value {value} < High Threshold {high_threshold}")
        elif moderate_threshold is not None and value < moderate_threshold:
            risk_level = 1  # Moderate risk
            # st.write(f"Moderate Risk: Value {value} < Moderate Threshold {moderate_threshold}")
    
    elif condition_type == "outside_range":
        # Ensure we have a valid range for moderate and high thresholds
        if (isinstance(moderate_threshold, list) and len(moderate_threshold) == 2 and 
            isinstance(high_threshold, list) and len(high_threshold) == 2):
            
            mod_lower, mod_upper = moderate_threshold
            high_lower, high_upper = high_threshold
            
            # Check for high risk first
            if value < high_lower or value > high_upper:
                risk_level = 2  # High risk
                # st.write(f"High Risk: Value {value} outside high threshold range [{high_lower}, {high_upper}]")
            
            # Then check for moderate risk
            elif value < mod_lower or value > mod_upper:
                risk_level = 1  # Moderate risk
                # st.write(f"Moderate Risk: Value {value} outside moderate threshold range [{mod_lower}, {mod_upper}]")
    
    # st.write(f"Determined Risk Level: {risk_level}")
    return risk_level

def get_risk(num):
    if num == 2:
        return 'high'
    elif num == 1:
        return 'moderate'
    else:
        return 'safe'
    
@st.cache_resource
def build_model_aggregated_dataset(df):

    # Convert categorical features to appropriate format
    df['SiteID'] = df['SiteID'].astype('category')
    df['Month'] = df['Month'].astype('category')


    # Check the range of years in your dataset
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    unique_years = sorted(df['Year'].unique())

    # st.write(f"Years in dataset: {unique_years}")
    # st.write((f"Range: {min_year} to {max_year} ({len(unique_years)} years)")

    # Calculate a good split point (e.g., use the last 20-25% of years for testing)
    # num_years = len(unique_years)
    # split_index = int(num_years * 0.8)  # Use 80% of years for training
    # split_year = unique_years[split_index]

    # print(f"Suggested split: Train on {unique_years[:split_index]} (years before {split_year})")
    # print(f"                 Test on {unique_years[split_index:]} (years {split_year} and after)")

    # # Create the train/test split
    # train_data = df[df['Year'] < split_year]
    # test_data = df[df['Year'] >= split_year]

    # print(f"Training data: {len(train_data)} samples from years {train_data['Year'].min()}-{train_data['Year'].max()}")
    # print(f"Testing data: {len(test_data)} samples from years {test_data['Year'].min()}-{test_data['Year'].max()}")

    # Alternative approach for imbalanced years: split by percentage of total data
    df_sorted = df.sort_values(['Year', 'Month'])
    train_size = 0.8
    train_data = df_sorted.iloc[:int(len(df_sorted) * train_size)]
    test_data = df_sorted.iloc[int(len(df_sorted) * train_size):]
    # st.write((f"Alternative - Training data: years {train_data['Year'].min()}-{train_data['Year'].max()}")
    # st.write((f"Alternative - Testing data: years {test_data['Year'].min()}-{test_data['Year'].max()}")

    # Create feature and target variables for training
    X_train = train_data.drop('risk_level', axis=1)  # Remove target column
    y_train = train_data['risk_level']               # Just the target column

    # Create feature and target variables for testing
    X_test = test_data.drop('risk_level', axis=1)
    y_test = test_data['risk_level']

    # Specify categorical features for CatBoost
    categorical_features = ['SiteID', 'Month']  # Add any other categorical columns

    # Now train the model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        cat_features=categorical_features,
        random_seed=42
    )

    # Fit the model
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100
    )

    # Evaluate performance
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_test)
    # st.write(classification_report(y_test, y_pred))
    # st.write(confusion_matrix(y_test, y_pred))

    return model


def aggregate_risk_by_date(df):

    # First let's make sure we have a single date column to work with
    if 'Date' not in df.columns:
        # Create Date from Year, Month, Day if needed
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # First ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract Year, Month, and Day components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Now aggregate to get the maximum risk level for each site and date
    aggregated_df = df.groupby(['SiteID', 'Date']).agg({
        'risk_level': 'max',  # Take highest risk from any indicator
        'Value': ['mean', 'min', 'max'],  # Aggregate value metrics
        # Include other columns you want to keep
        'Year': 'first',
        'Month': 'first',
        'Day': 'first'
    }).reset_index()

    # Flatten multi-level column names
    aggregated_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in aggregated_df.columns]
    # st.write('aggregate df columns are ', aggregated_df.columns)
    # st.write(f"Original data: {len(df)} rows")
    # st.write(f"Aggregated data: {len(aggregated_df)} rows")
    # st.write(f"Unique site-date combinations: {df.groupby(['SiteID', 'Date']).ngroups}")

    # Verify our aggregation worked as expected
    # st.write("Risk level distribution before aggregation:")
    # st.write(df['risk_level'].value_counts())
    # st.write("aggregated df columns:", aggregated_df.columns.tolist())
    aggregated_df = aggregated_df.rename(columns={'risk_level_max': 'risk_level'})
    aggregated_df = aggregated_df.rename(columns={'Day_first': 'Day'})
    aggregated_df = aggregated_df.rename(columns={'Month_first': 'Month'})
    aggregated_df = aggregated_df.rename(columns={'Year_first': 'Year'})
    # st.write("Risk level distribution after taking max per site-date:")

    # st.write(aggregated_df['risk_level'].value_counts())
    return aggregated_df


def predict_risk_for_site_month(model, df, site_id, target_month, current_year):
    """
    Predict risk level using most recent historical data for the same site and month.
    
    Parameters:
    model: Trained prediction model
    df: aggregated DataFrame
    site_id: Target SiteID to predict for
    target_month: Month to predict for (1-12)
    current_year: Current year (to avoid using future data)
    
    Returns:
    Predicted risk level
    """
    # Filter for the specific site and month (from previous years)
    historical_data = df[(df['SiteID'] == site_id) & 
                         (df['Month'] == target_month) & 
                         (df['Year'] < current_year)]
    
    if len(historical_data) == 0:
        st.write(f"No historical data found for SiteID={site_id}, Month={target_month}")
        return None
    
    # Get the most recent year's data for this site and month
    most_recent_year = historical_data['Year'].max()
    most_recent_data = historical_data[historical_data['Year'] == most_recent_year]
    
    # Create input data for the model using the same structure as training data
    # But with Year updated to current year
    input_data = most_recent_data.copy()
    input_data['Year'] = current_year
    input_data['Month'] = target_month
    input_data['Date'] = pd.to_datetime([f"{current_year}-{target_month}-15"])[0]
    # Select only the columns used by the model
    model_features = [col for col in input_data.columns if col != 'risk_level']
    # st.write('model features:  ', input_data[model_features])
    
    # Make prediction
    predicted_risk = model.predict(input_data[model_features])[0]
    
    return predicted_risk

def aggregate_for_calendar_year(site_id, df):
    # Assuming your dataframe is called 'df' with columns 'siteID', 'date', and 'risk_level'
    site_data = df[df['SiteID'] == site_id].copy()
    # First, make sure your date column is datetime
    site_data['Date'] = pd.to_datetime(df['Date'])

    # Group by siteID, month, and day, then calculate mean risk level
    calendar_df = site_data.groupby(['SiteID', 'Month', 'Day'])['risk_level'].mean().reset_index()

    # Round to integers if your risk levels should be whole numbers
    # If you want to keep decimals, remove this line
    calendar_df['risk_level'] = calendar_df['risk_level'].round().astype(int)

    # Create a date column for visualization (using a reference year, e.g., 2025)
    # st.dataframe(calendar_df)
    # calendar_df['Date'] = pd.to_datetime({
    #     'year': 2024,
    #     'month': calendar_df['Month'],
    #     'day': calendar_df['Day']
    # })
    
    # Use apply to create dates properly
    calendar_df['Date'] = calendar_df.apply(
        lambda row: pd.Timestamp(year=2025, month=row['Month'], day=row['Day']), axis=1
    ) 
    return calendar_df

def heatmap_risk_by_date(site_id, site_data):
    # Create heatmap of risk by day and month
    
    try:
        # Convert day-month to a date string for better sorting
        # site_data['MonthDay'] = site_data['DateObj'].dt.strftime('%m-%d')
        
        # Count occurrences of each risk level by day of year
        daily_data = site_data.copy()
        # st.dataframe(daily_data)
    

        if daily_data['risk_level'].isnull().any():
            st.write("Warning: There are NaN values in 'risk_level' column")
        # Create matrix for heatmap (months x days)
        # Initialize with NaN
        calendar_data = np.full((12, 31), np.nan)
        
        # Fill in with average risk values
        for m in range(1, 13):
            month_data = daily_data[daily_data['Month'] == m]
            for d in range(1, 32):
                day_data = month_data[month_data['Day'] == d]
                if len(day_data) > 0:
                    calendar_data[m-1, d-1] = day_data['risk_level'].mean()
        
        # Create calendar heatmap
        fig = plt.figure(figsize=(14, 8))
        
        # Custom colormap: green for safe, yellow for moderate, red for high
        cmap = plt.cm.colors.ListedColormap(['green', 'yellow', 'red'])
        bounds = [0, 0.67, 1.33, 2]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create heatmap
        heatmap = plt.pcolormesh(calendar_data.T, cmap=cmap, norm=norm)
        # Set labels
        plt.yticks(np.arange(0.5, 31.5), np.arange(1, 32))
        plt.xticks(np.arange(0.5, 12.5), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.title(f'Historical Calendar View of Risk Levels for {site_id}', fontsize=18)
        plt.ylabel('Day of Month')
        
        # Add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_ticks([0.33, 1, 1.67])
        cbar.set_ticklabels(['Safe', 'Moderate Risk', 'High Risk'])
        
        plt.tight_layout()
        st.pyplot(fig)  # Return the monthly figure for displayylabel('Number of Days')
    except Exception as e:
        st.write(f"Error creating calendar visualization: {e}")
    
 

st.title("New Zealand Lakes Water Quality")
df = read_data()

df['risk_level'] = df.apply(determine_risk_level, axis=1)
aggregated_df = aggregate_risk_by_date(df)
model = build_model_aggregated_dataset(aggregated_df)

filtered_sites = []
for site in df['SiteID'].unique():
    # Check if the site ID is not a string that consists only of digits
    if not (isinstance(site, str) and site.isdigit()):
        filtered_sites.append(site)
site = st.sidebar.selectbox('Lakes', sorted(filtered_sites))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month = st.sidebar.selectbox('Month', months)
current_year = datetime.now().year

# Create a mapping of risk levels to their colors
risk_color_map = {
    "safe": "green",
    "moderate": "yellow", 
    "high": "red",
    "unknown": "gray"
}

# Use HTML for both centering and dynamic coloring

if st.sidebar.button("Run Prediction"):
    predicted_risk = predict_risk_for_site_month(model, aggregated_df, site, months.index(month)+1, current_year)
    risk_level = get_risk(predicted_risk)
    # Generate the colored text using the mapping
    my_color = risk_color_map.get(risk_level, "black")  # Default to black if not found
    st.subheader(f"Predicted risk level for {site} in {month}/{current_year} is ") 
    st.markdown(f"<div style='text-align: center; color: {my_color}; font-size: 46px; font-weight: bold'>{risk_level.replace('_', ' ').title()}</div>", unsafe_allow_html=True)

 #   st.markdown(f"<div style='text-align: center; color: {color}; font-size: 46px; font-weight: bold'>{risk_level}</div>", unsafe_allow_html=True)                      
    heatmap_risk_by_date(site, aggregate_for_calendar_year(site, aggregated_df))
    
else:
        st.write("Select your lake site and month of year in the sidebar and click 'Run Prediction' to see the risk level")