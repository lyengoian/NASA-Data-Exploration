"""
Lilit Yengoian
Final Project
"""

import matplotlib
import numpy as numpy
from sklearn.linear_model import LinearRegression
from flask import Flask, redirect, render_template, request, session, url_for
import os
import sqlite3 as sl
import pandas as pd
from matplotlib.figure import Figure
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')

app = Flask(__name__)

'''Reading in csv file into a dataframe and making a database out of the data in the dataframe'''
db = "nasa_data.db"  # making new database
df = pd.read_csv("PSCompPars_2022.04.30_17.34.07.csv")  # reading into csv into a dataframe
df = df.filter(['pl_name', 'disc_year', 'pl_bmasse', 'disc_facility'])  # only keeping these columns
conn = sl.connect('nasa_data.db')
cursor = conn.cursor()
# cursor.execute('''CREATE TABLE nasa (pl_name, disc_year, pl_bmasse, disc_facility)''') # ONLY RUN THIS ONCE
for row in df.itertuples():  # for every row in the dataframe
    # add in the data for each row into the database
    cursor.execute('''
                INSERT INTO nasa (pl_name, disc_year, pl_bmasse, disc_facility)
                VALUES (?,?,?, ?) 
                ''',
                   (row.pl_name,
                    row.disc_year,
                    row.pl_bmasse,
                    row.disc_facility)
                   )
conn.commit() # stored my data in the server as a database

'''This is where the home page is. The user can choose from three options, and based on what they select, 
they will be directed to another webpage '''


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        selected_Value = request.form["option"]
        if selected_Value == 'masses':
            return render_template("masses.html", df=db_get_stations(), message="Pick a station")
        if selected_Value == "visualize":
            return render_template('visualize.html', df=db_get_stations(), message="Graph")
        if selected_Value == 'prediction':
            return render_template('prediction.html', df=db_get_stations(), message="Prediction")
    else:
        return render_template('home.html')


'''Here, the user will be presented with a list of all the facilities to choose from. They will be prompted to enter 
a facility name. This function takes the data (facility name) and stores it and passes it into the station_masses 
function '''


@app.route("/masses", methods=['GET', 'POST'])
def masses():
    if request.method == "POST":
        selected_station = request.form["text"]
        session['selected_station'] = selected_station
        return redirect(url_for('station_masses', st=session['selected_station']))
    return render_template("masses.html", df=db_get_stations())


'''This function first checks if the user entered a valid facility name in the previous page. Then, it provides the 
average mass of planets found by that facility. It does this by passing in the dataframe and facility name into the 
db_get_station_mass function '''


@app.route('/station/<st>', methods=["GET"])
def station_masses(st):
    if db_check_facility(st):
        x = db_get_station_mass(st, df)
        for key in x:
            y = x[key]
        return f"<h1>{st}:</h1>" \
               f"<h3>The average mass of all the planets found is {y} Earth Mass</h3>"
    else:
        return redirect(url_for('masses',
                                df=db_get_stations()))
        # if the user doesn't enter a valid facility name, they get redirected to the previous page to reenter a name


'''Here, the user will be presented with a list of all the facilities to choose from. They will be prompted to enter 
a facility name. This function takes the data (facility name) and stores it and passes it into the station_masses_vis 
function '''


@app.route("/visualize", methods=['GET', 'POST'])
def visualize():
    if request.method == "POST":
        selected_station = request.form["text"]
        session['selected_station'] = selected_station
        return redirect(url_for('station_masses_vis', st=session['selected_station']))
    return render_template("visualize.html", df=db_get_stations())


'''This function first checks if the user entered a valid facility name in the previous page. Then, it provides a bar 
graph of the masses of all the planets found by that facility. It does this by passing in the dataframe and facility 
name into the db_get_visuals function '''


@app.route('/visualize/<st>', methods=["GET"])
def station_masses_vis(st):
    if db_check_facility(st):
        db_get_visuals(st, df)
        # the plot.png comes from the plot_png function that takes in the figure made by db_get_visuals(st, df)
        return f"<h1>{st}</h1>  <img src='/plot.png' alt='my plot'> <p>Note: Existing Data"
    else:
        return redirect(url_for('visualize',
                                df=db_get_stations()))
        # if the user doesn't enter a valid facility name, they get redirected to the previous page to reenter a name


'''This function prompts the user to enter a future year. It then passes that year into the st_prediction function'''


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        year1 = request.form["year1"]
        session['year1'] = year1
        return redirect(url_for('st_prediction', year1=session['year1']))
    return render_template("prediction.html", df=db_get_stations())


'''This function displays the predicted average mass of all exoplanets that will be found in the given year. It gets 
this info by passing in the year entered by the user into the db_get_prediction function '''


@app.route('/prediction/<year1>', methods=["GET"])
def st_prediction(year1):
    x = db_get_prediction(year1)
    print(type(x))
    return f"<h1>Predicted average mass of exoplanets that will be found in {year1}: <h1>{x[0]} Earth Mass</h1> </h1> "\
           f"<img src='/plotml.png' alt='my plot'> "


'''This function takes in the station name entered by the user and passes it to the db_get_visuals function to get a 
figure fig. '''


@app.route('/plot.png')
def plot_png():
    st = session['selected_station']
    fig = db_get_visuals(st, df)
    output_file = io.BytesIO()  # BytesIO puts a wrapper around the output_file to behave like a file
    FigureCanvas(fig).print_png(output_file)  # matplotlib function to write the figure to a png file
    return Response(output_file.getvalue(), mimetype='image/png')  # mimetype makes png with value of output_file (fig)



@app.route('/plotml.png')
def plotml_png():
    year1 = session['year1']
    return_list = db_get_prediction(year1)
    avg_mass = return_list[0]
    fig = return_list[1]
    output_file = io.BytesIO()  # BytesIO puts a wrapper around the output_file to behave like a file
    FigureCanvas(fig).print_png(output_file)  # matplotlib function to write the figure to a png file
    return Response(output_file.getvalue(), mimetype='image/png')  # mimetype makes png with value of output_file (fig)


'''This is where I do ML -- cleaning the data, training the model, getting a prediction, etc. This function returns 
back the predicted avg mass. I am doing this to find the average mass of ALL planets found by ALL stations in a 
future year, based on prior data. Because my data was floats, I was only able to do Linear Regression although I 
attempted StandardScaler and MLPClassifier '''


def db_get_prediction(year1):
    df = pd.read_csv('PSCompPars_2022.04.30_17.34.07.csv')  # reading in the dataframe from scratch
    df = df.drop(df[df['pl_bmasse'] > 17000].index)  # taking out the very obvious outliers from the data
    df['pl_bmasse'] = df['pl_bmasse'].fillna(df['pl_bmasse'].mean())
    # filling in missing mass values with the average of all masses
    year_list = df['disc_year'].unique()  # creating a year list so that I can group by year
    dataset_grouped = df.groupby("disc_year")  # grouping by year
    avg = []  # list to hold the masses
    for year in year_list:
        avg.append(dataset_grouped.get_group(year)["pl_bmasse"].mean())
        # adding in the average masses for each year into the avg list
    planet_list = dict(zip(year_list, avg))  # creating a dictionary of year_list : avg values
    df = pd.DataFrame(list(planet_list.items()), columns=['disc_year', 'pl_bmasse'])
    # making a new dataframe out of the dictionary's keys and values, and setting the column names to match the original
    # data. Here, years are all unique and masses are the average masses for that year
    df = df.drop(df[df['pl_bmasse'] == 3397.602700].index)  # taking out another weird datapoint
    X = df.iloc[:, :-1].values  # all rows for all but the last column (only disc_year, but not pl_bmasse column))
    y = df.iloc[:, 1].values  # taking all the rows for the second column (pl_bmasse)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    model = LinearRegression()  # making linearRegression model
    model.fit(X_train, y_train)  # fitting the data to the model
    future_year = [[year1]]  # taking in the year1 entered by user and using it as the future_year variable
    predicted_avg_mass = model.predict(future_year)
    # predicted average mass is equal to the value predicted by the model when x = future_year
    numpy.append(year_list, year1)
    numpy.append(avg, predicted_avg_mass)
    x = year_list
    y = avg
    fig = Figure()  # making figure object
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    legend_name = ['All years']
    ax.set(xlabel="YEAR", ylabel="Average mass (Earth Mass)",
           title="Average masses of planets found")
    ax.legend(legend_name, prop={'size': 10})
    return [predicted_avg_mass, fig]  # return predicted avg mass


''' This is a function for getting list of facilities/stations to display on the webpage for the user to choose from'''


def db_get_stations():
    facility_list = df['disc_facility'].unique()  # get list of all facilities
    return facility_list


''' This is where the station avg mass is calculates '''


def db_get_station_mass(st, df):
    facility_list = df['disc_facility'].unique()  # getting list of the facilities
    df_grouped = df.groupby("disc_facility")  # grouping data facility
    avg = []
    for facility in facility_list:
        if facility == st:
            avg.append(df_grouped.get_group(facility)["pl_bmasse"].mean())  # add average for that facility into avg[]
    planet_list = dict(zip(facility_list, avg))  # make a dict with key = facility, value = avg mass
    return planet_list  # return the dictionary


''' This function makes the bar graph and returns the figure '''


def db_get_visuals(st, df):
    df = df.drop(df[df['disc_facility'] != st].index)  # dropping all data that doesn't correspond to the chose facility
    fig = Figure()  # making figure object
    ax = fig.add_subplot(1, 1, 1)
    year_list = df['disc_year'].unique()  # getting list of unique years
    df_grouped = df.groupby("disc_year")  # grouping data by year
    avg = []
    for year in year_list:
        avg.append(df_grouped.get_group(year)["pl_bmasse"].mean())  # getting the average mass for every year
    planet_list = dict(zip(year_list, avg))  # putting all the data in a dictionary
    x = planet_list.keys()
    y = planet_list.values()
    ax.bar(x, y)  # plotting x vs y (year vs average mass)
    legend_name = [st]
    ax.set(xlabel="YEAR", ylabel="Average mass (Earth Mass)",
           title="Average masses of planets found by " + st)
    ax.legend(legend_name, prop={'size': 10})
    return fig  # returning the figure to a function to generate the png


'''This function checks if the user entered a valid station name'''


def db_check_facility(st):
    facility_list = df["disc_facility"].unique()  # make list of unique station names
    if st in facility_list:  # if the named entered by the user is in the list, return true
        return True
    else:
        return False


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(host="0.0.0.0", port=8080, debug=True)
    app.run(debug=True)
