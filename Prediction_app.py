from os import error
from flask import Flask, render_template, request
#from datacleaning import *
import pickle
import re
import pandas as pd
#
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        def clean1():
            dic = request.form

            col = list(dic.keys())
            row = [list(dic.values())]

            df = pd.DataFrame(row,columns=col)
            df['Duration'] = df['hours'] + '.' + df['min']

            df = df.astype({'Duration':float})

            df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])

            df = df.drop(columns = ['hours','min'],axis=1)
            
            return df

        def clean2(test):    
            test['Month']= [i.month for i in test['Date_of_Journey']]
            #Days
            test['Day']= [i.day for i in test['Date_of_Journey']]

            test = pd.get_dummies(test,columns=['Airline','Source', 'Destination','Additional_Info'])

            test = test.drop(columns = ['Date_of_Journey'],axis=1)

            def cleanDeptime(text):
                    text = re.sub(':','.',text)
                    return text

            test['Dep_Time'] = test['Dep_Time'].apply(cleanDeptime)

            test = test.astype({'Duration':float})
            test = test.astype({'Dep_Time':float})

            val = test.values.tolist()[0]

            col =test.columns.tolist()

            testdic= {}
            for i,j in list(zip(col,val)):
                testdic.update({i:j})

            coldic = {'Dep_Time': 0,
            'Duration': 0,
            'Total_Stops': 0,
            'Airline_Air Asia': 0,
            'Airline_Air India': 0,
            'Airline_GoAir': 0,
            'Airline_IndiGo': 0,
            'Airline_Jet Airways': 0,
            'Airline_Jet Airways Business': 0,
            'Airline_Multiple carriers': 0,
            'Airline_Multiple carriers Premium economy': 0,
            'Airline_SpiceJet': 0,
            'Airline_Trujet': 0,
            'Airline_Vistara': 0,
            'Airline_Vistara Premium economy': 0,
            'Source_Banglore': 0,
            'Source_Chennai': 0,
            'Source_Delhi': 0,
            'Source_Kolkata': 0,
            'Source_Mumbai': 0,
            'Destination_Banglore': 0,
            'Destination_Cochin': 0,
            'Destination_Delhi': 0,
            'Destination_Hyderabad': 0,
            'Destination_Kolkata': 0,
            'Destination_New Delhi': 0,
            'Additional_Info_1 Long layover': 0,
            'Additional_Info_1 Short layover': 0,
            'Additional_Info_2 Long layover': 0,
            'Additional_Info_Business class': 0,
            'Additional_Info_Change airports': 0,
            'Additional_Info_In-flight meal not included': 0,
            'Additional_Info_No Info': 0,
            'Additional_Info_No check-in baggage included': 0,
            'Additional_Info_No info': 0,
            'Additional_Info_Red-eye flight': 0,
            'Month': 0,
            'Day': 0}

            coldic.update(testdic)

            col = list(coldic.keys())
            row = [list(coldic.values())]
            newdf = pd.DataFrame(row,columns=col)

            return newdf
        def output_pred():
            Test = clean1()
            newdf = clean2(Test)
            forest = open('flight_model.pkl','rb')
            model = pickle.load(forest)
            pred_price = model.predict(newdf)
            output = round(pred_price[0],2)
            output = str(output)
            return output
        output = output_pred()
    return render_template('prediction.html',P = output)

@app.route('/pre')  
def pre():
    return 'Surcessfully failed'



if __name__ == "__main__":
    app.run()


