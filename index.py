# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set visualization style
plt.style.use('ggplot')
%matplotlib inline

# Load the dataset (replace with your local path if needed)
try:
    df = pd.read_csv('owid-covid-data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please download the dataset from Our World in Data.")

    # Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
display(df.head())

print("\nColumns in the dataset:")
print(df.columns.tolist())

print("\nMissing values summary:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Filter for selected countries
countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'Kenya', 'South Africa']
df_filtered = df[df['location'].isin(countries)].copy()

# Handle missing values for key metrics
cols_to_fill = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
               'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
df_filtered[cols_to_fill] = df_filtered[cols_to_fill].fillna(0)

# Calculate death rate (with handling for division by zero)
df_filtered['death_rate'] = np.where(df_filtered['total_cases'] > 0, 
                                    df_filtered['total_deaths'] / df_filtered['total_cases'], 
                                    0)

# Create a month-year column for aggregation
df_filtered['month_year'] = df_filtered['date'].dt.to_period('M')

print("\nData after cleaning:")
print(f"Time period: {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")
print(f"Countries: {df_filtered['location'].unique().tolist()}")

# Aggregate global data (sum across all countries)
global_df = df_filtered.groupby('date')[['new_cases', 'new_deaths']].sum().reset_index()

plt.figure(figsize=(14, 6))
plt.plot(global_df['date'], global_df['new_cases'], label='New Cases')
plt.plot(global_df['date'], global_df['new_deaths'], label='New Deaths', color='red')
plt.title('Global Daily New COVID-19 Cases and Deaths')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

# Get latest data for each country
latest_data = df_filtered.sort_values('date').groupby('location').last().reset_index()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='total_cases', y='location', data=latest_data.sort_values('total_cases', ascending=False))
plt.title('Total COVID-19 Cases by Country')
plt.xlabel('Total Cases')
plt.ylabel('Country')

plt.subplot(1, 2, 2)
sns.barplot(x='total_deaths', y='location', data=latest_data.sort_values('total_deaths', ascending=False))
plt.title('Total COVID-19 Deaths by Country')
plt.xlabel('Total Deaths')
plt.ylabel('')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='death_rate', y='location', 
            data=latest_data.sort_values('death_rate', ascending=False))
plt.title('COVID-19 Death Rate by Country (Total Deaths / Total Cases)')
plt.xlabel('Death Rate')
plt.ylabel('Country')
plt.show()

# Plot vaccination progress over time
plt.figure(figsize=(14, 8))
for country in countries:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['people_fully_vaccinated_per_hundred'], 
             label=country)

plt.title('COVID-19 Vaccination Progress (Fully Vaccinated per 100 People)')
plt.xlabel('Date')
plt.ylabel('Percentage Fully Vaccinated')
plt.legend()
plt.grid(True)
plt.show()

# Latest vaccination status
plt.figure(figsize=(10, 6))
sns.barplot(x='people_fully_vaccinated_per_hundred', y='location', 
            data=latest_data.sort_values('people_fully_vaccinated_per_hundred', ascending=False))
plt.title('Percentage of Population Fully Vaccinated')
plt.xlabel('Percentage Fully Vaccinated')
plt.ylabel('Country')
plt.show()

# Prepare data for choropleth
latest_global = df.sort_values('date').groupby('location').last().reset_index()

# Create choropleth map of total cases per million
fig = px.choropleth(latest_global, 
                    locations="iso_code",
                    color="total_cases_per_million",
                    hover_name="location",
                    hover_data=["total_cases", "total_deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Global COVID-19 Cases per Million People")
fig.show()

# Export the cleaned data for future use
df_filtered.to_csv('cleaned_covid_data.csv', index=False)
print("Analysis complete. Cleaned data saved to 'cleaned_covid_data.csv'")