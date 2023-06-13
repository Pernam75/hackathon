import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

# Load data
df = pd.read_csv('df.csv')
country_list = df['Country'].unique().tolist()
country_list_without_france = country_list.copy()
country_list_without_france.remove('France')

# Function to display data
def display_data():
    st.write(df.head(100))

# Function to display descriptive statistics
def display_statistics():
    st.write(df.describe())
    # Plot the co2 emission of the countries of the df on a map for 2000
    fig = px.choropleth(df[df['Year'] == 2000], locations="Country", locationmode='country names', color="CO2e_Emissions", hover_name="Country", color_continuous_scale=px.colors.sequential.Plasma, title='CO2 emission of the countries of the df on a map for 2000')
    st.plotly_chart(fig)

    # Plot the co2 emission of the countries of the df on a map for 2021
    fig = px.choropleth(df[df['Year'] == 2021], locations="Country", locationmode='country names', color="CO2e_Emissions", hover_name="Country", color_continuous_scale=px.colors.sequential.Plasma, title='CO2 emission of the countries of the df on a map for 2021')
    st.plotly_chart(fig)

def analysis_emissions_consumption():
    options = st.multiselect(
    'Choose the country of analysis:',
    country_list_without_france)

    st.write('Choose the country of analysis:', options)

    choose_country = ['France']
    if options != []:
        choose_country = choose_country + options

    df_choose = df[df['Country'].isin(choose_country)]
    st.write(df_choose.head())

    fig = plt.figure(figsize=(20, 10))
    fig = px.line(df_choose, x="Year", y="CO2e_Emissions", color='Country')
    st.plotly_chart(fig)

    energy_sources = ['Oil Consumption - EJ', 'Gas Consumption - EJ',
                  'Coal Consumption - EJ', 'Nuclear Consumption - EJ',
                  'Hydro Consumption - EJ', 'Solar Consumption - EJ',
                  'Wind Consumption - EJ', 'Geo Biomass Other - EJ',
                  'Biofuels consumption - EJ']

    df_choose_2000 = df_choose[df_choose['Year'] == 2000]
    df_choose_2000[energy_sources] = df_choose_2000[energy_sources].div(df_choose_2000[energy_sources].sum(axis=1), axis=0)
    energy_by_country = df_choose_2000.pivot_table(values=energy_sources, index='Country')

    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(energy_by_country, cmap='YlGnBu')
    plt.xlabel('Energy Sources')
    plt.ylabel('Country')
    plt.title('Energy Consumption by Country and Source - 2000')
    st.pyplot(fig)

    df_choose_2021 = df_choose[df_choose['Year'] == 2021]
    df_choose_2021[energy_sources] = df_choose_2021[energy_sources].div(df_choose_2021[energy_sources].sum(axis=1), axis=0)
    energy_by_country = df_choose_2021.pivot_table(values=energy_sources, index='Country')

    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(energy_by_country, cmap='YlGnBu')
    plt.xlabel('Energy Sources')
    plt.ylabel('Country')
    plt.title('Energy Consumption by Country and Source - 2021')
    st.pyplot(fig)

    # Plot the evolution of the energy consumption in a subplots per country
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    fig.suptitle('Energy Consumption by Country and Source - 2000')
    sns.barplot(ax=axes[0, 0], x='Country', y='Oil Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[0, 1], x='Country', y='Gas Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[1, 0], x='Country', y='Coal Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[1, 1], x='Country', y='Nuclear Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[2, 0], x='Country', y='Hydro Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[2, 1], x='Country', y='Solar Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[0, 2], x='Country', y='Wind Consumption - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[1, 2], x='Country', y='Geo Biomass Other - EJ', data=df_choose_2000)
    sns.barplot(ax=axes[2, 2], x='Country', y='Biofuels consumption - EJ', data=df_choose_2000)
    st.pyplot(fig)

    # Plot the evolution of the energy consumption in a subplots per country
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    fig.suptitle('Energy Consumption by Country and Source - 2021')
    sns.barplot(ax=axes[0, 0], x='Country', y='Oil Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[0, 1], x='Country', y='Gas Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[1, 0], x='Country', y='Coal Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[1, 1], x='Country', y='Nuclear Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[2, 0], x='Country', y='Hydro Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[2, 1], x='Country', y='Solar Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[0, 2], x='Country', y='Wind Consumption - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[1, 2], x='Country', y='Geo Biomass Other - EJ', data=df_choose_2021)
    sns.barplot(ax=axes[2, 2], x='Country', y='Biofuels consumption - EJ', data=df_choose_2021)
    st.pyplot(fig)

def prediction_co2_emissions_2022_world():
    df_predict = pd.DataFrame(columns=['Country', 'CO2e_Emissions','Pourcentage'])
    list_country = df["Country"].unique()
    for country in list_country:
        df_count = df[df['Country'] == country]
        df_count = df_count.drop(columns=['Country'])
        X = df_count[['Year', 'Oil Consumption - EJ', 'Gas Consumption - EJ',
        'Coal Consumption - EJ', 'Nuclear Consumption - EJ',
        'Hydro Consumption - EJ', 'Solar Consumption - EJ',
        'Wind Consumption - EJ', 'Geo Biomass Other - EJ',
        'Biofuels consumption - EJ']]
        y = df_count['CO2e_Emissions']
        X_train = X[X['Year'] < 2019]
        X_test = X[X['Year'] >= 2019]
        y_train = y[X['Year'] < 2019]
        y_test = y[X['Year'] >= 2019]

        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)
        X_2022 = df_count[df_count['Year'] == 2021][['Year', 'Oil Consumption - EJ', 'Gas Consumption - EJ',
                                        'Coal Consumption - EJ', 'Nuclear Consumption - EJ',
                                        'Hydro Consumption - EJ', 'Solar Consumption - EJ', 
                                        'Wind Consumption - EJ', 'Geo Biomass Other - EJ',
                                        'Biofuels consumption - EJ']]
        y_2022 = linreg.predict(X_2022)
        val = df_count[df_count['Year'] == 2021]['CO2e_Emissions'].values
        pourcent_predict= y_2022[0]/val[0]
        df_predict = df_predict.append({'Country': country, 'CO2e_Emissions': y_2022[0], 'Pourcentage': pourcent_predict}, ignore_index=True)

    df_2022_pourcent = df_predict.copy()
    df_2022_pourcent["Pourcentage"] = df_2022_pourcent["Pourcentage"] -1
    df_2022_pourcent["Pourcentage"] = df_2022_pourcent["Pourcentage"] * 100
    df_2022_pourcent.drop(columns=['CO2e_Emissions'], inplace=True)
    df_2022_pourcent = df_2022_pourcent[df_2022_pourcent['Country'] != 'Vietnam']
    df_2022_pourcent = df_2022_pourcent[df_2022_pourcent['Country'] != 'Colombia']

    df_2022_emission = df_predict.copy()
    df_2022_emission.drop(columns=['Pourcentage'], inplace=True)
    df_2022_emission = df_2022_emission[df_2022_emission['Country'] != 'Vietnam']
    df_2022_emission = df_2022_emission[df_2022_emission['Country'] != 'Colombia']


    fig = px.choropleth(df_2022_emission, locations="Country", locationmode='country names', color="CO2e_Emissions", hover_name="Country", color_continuous_scale=px.colors.sequential.Plasma, title='CO2 emission of the countries on a map for 2022')
    st.plotly_chart(fig)

    # Plot the pourcentage of CO2 emission for 2022 on a map
    fig = px.choropleth(df_2022_pourcent, locations="Country", locationmode='country names', color="Pourcentage", hover_name="Country", color_continuous_scale=px.colors.sequential.Plasma, title='Pourcentage of CO2 emission for 2022')
    st.plotly_chart(fig)

def prediction_co2_emissions_2022_country():
    option = st.selectbox('Select the country where you want the prediction',country_list)
    print(option)
    if option:
        df_fr = df[df['Country'] == option]
        print(df_fr)
        X = df_fr[['Year', 'Oil Consumption - EJ', 'Gas Consumption - EJ',
            'Coal Consumption - EJ', 'Nuclear Consumption - EJ',
            'Hydro Consumption - EJ', 'Solar Consumption - EJ',
            'Wind Consumption - EJ', 'Geo Biomass Other - EJ',
            'Biofuels consumption - EJ']]
        y = df_fr['CO2e_Emissions']

        X_train = X[X['Year'] < 2019]
        X_test = X[X['Year'] >= 2019]
        y_train = y[X['Year'] < 2019]
        y_test = y[X['Year'] >= 2019]

        # Create the model
        linreg = LinearRegression()

        # Train the model
        linreg.fit(X_train, y_train)

        # Make predictions
        y_pred = linreg.predict(X_test)

        # Evaluate the model
        st.write('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        st.write('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

        # Predict the CO2 emission for 2022
        X_2022 = df_fr[df_fr['Year'] == 2021][['Year', 'Oil Consumption - EJ', 'Gas Consumption - EJ',
                                                'Coal Consumption - EJ', 'Nuclear Consumption - EJ',
                                                'Hydro Consumption - EJ', 'Solar Consumption - EJ',
                                                'Wind Consumption - EJ', 'Geo Biomass Other - EJ',
                                                'Biofuels consumption - EJ']]
        y_2022 = linreg.predict(X_2022)
        st.write('Predicted CO2 emission for 2022: %.2f' % y_2022)
        val = df_fr[df_fr['Year'] == 2021]['CO2e_Emissions'].values
        pourcent_predict= y_2022[0]/val[0]
        print(pourcent_predict)
        st.write('Pourcentage of variation between 2021 and our prediction for 2022: ', pourcent_predict)

        # Plot the evolution of the energy consumption for France with 2022 prediction
        fig = plt.figure(figsize=(20, 10))
        fig = px.line(df_fr, x="Year", y="CO2e_Emissions", color='Country')
        fig.add_scatter(x=[2022], y=y_2022, mode='markers', name='2022 prediction')
        st.plotly_chart(fig)

# Main function to run the Streamlit app
def main():    
    # Sidebar options
    option = st.sidebar.selectbox('Select an option', ('Display Data', 'Display Statistics', 'Analysis of the emissions of CO2 in the world and consumption of energy', 'Prediction of CO2 emissions in 2022', 'Prediction of CO2 emissions in 2022 for a country', 'Forecasting'))
    
    # Display data or statistics based on the selected option
    if option == 'Display Data':
        st.title('Display the data')
        display_data()
    elif option == 'Display Statistics':
        st.title('Exploratory Data Analysis')
        display_statistics()
    elif option == 'Analysis of the emissions of CO2 in the world and consumption of energy':
        st.title('Analysis between France and choosen country')
        analysis_emissions_consumption()
    elif option == 'Prediction of CO2 emissions in 2022':
        st.title('Prediction of CO2 emission in 2022')
        prediction_co2_emissions_2022_world()
    elif  option == 'Prediction of CO2 emissions in 2022 for a country':
        st.title('Prediction of CO2 emission in 2022 for a country')
        prediction_co2_emissions_2022_country()
    elif  option == 'Forecasting':
        st.title('Forecasting the CO2 emission for France')
        image = Image.open('forecasting_france.png')
        st.image(image, caption='Foercasting the CO2 emission for France')
        
        


# Run the app
if __name__ == '__main__':
    main()