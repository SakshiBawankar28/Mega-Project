import os

import pandas as pd
import pygal
from flask import Flask,render_template,request
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import seaborn as sns

from tensorflow.keras.models import load_model


app = Flask(__name__)
app.config['upload_folder']=r'uploads'
global df
global path
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load',methods=["POST","GET"])
def load_data():
    if request.method=="POST":
        print('1111')
        files = request.files['file']
        print(files)
        filetype = os.path.splitext(files.filename)[1]
        if filetype == '.csv':
            print('111')
            path = os.path.join(app.config['upload_folder'],files.filename)
            files.save(path)
            print(path)
            return render_template('Load Data.html',msg='valid')
        else:
            return render_template('Load Data.html',msg= 'invalid')
    return render_template('Load Data.html')

@app.route('/preprocess')
def preprocess():
    file = os.listdir(app.config['upload_folder'])
    path =os.path.join(app.config['upload_folder'],file[0])
    df = pd.read_csv(path)
    print(df.head())
    df.isnull().sum()
    return render_template('Pre-process Data.html',msg = 'success')


@app.route('/viewdata',methods=["POST","GET"])
def view_data():
    file = os.listdir(app.config['upload_folder'])
    path = os.path.join(app.config['upload_folder'], file[0])
    df = pd.read_csv(path)
    df1 = df.sample(frac=0.3)
    df1 = df1[:100]
    print(df1)
    return render_template('view data.html',col_name = df1.columns, row_val=list(df1.values.tolist()))

@app.route('/model',methods=["POST","GET"])
def model():
    # global lascore, lpscore, lrscore
    # global nascore, npscore, nrscore
    # global aascore, apscore, arscore
    # global kascore, kpscore, krscore
    global accuracy4,recall4,precision4
    global accuracy1, recall1, precision1
    global accuracy3, recall3, precision3
    global accuracy2, recall2, precision2
    if request.method == "POST":
        model = int(request.form['selected'])
        file = os.listdir(app.config['upload_folder'])
        path = os.path.join(app.config['upload_folder'], file[0])
        df = pd.read_csv(path)
        df1 = df.sample(frac=0.3)
        X = df1.drop(['Time','Class'],axis = 1)
        y= df1.Class
        global train_test_split
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 10)
        if model == 1:
            lr = LogisticRegression(solver='sag')
            model1 = lr.fit(x_train,y_train)
            pred = model1.predict(x_test)
            accuracy1 = accuracy_score(y_test,pred)
            precision1= precision_score(y_test,pred)
            recall1 = recall_score(y_test,pred)
            return render_template('model.html',msg='accuracy',score =accuracy1,selected = 'LOGISTIC REGRESSION')
        elif model == 2:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # We are transforming data to numpy array to implementing with keras
            X_train = pd.np.array(X_train)
            X_test = pd.np.array(X_test)
            y_train = pd.np.array(y_train)
            y_test = pd.np.array(y_test)
            X_train.shape
            model = Sequential([
                Dense(units=20, input_dim=X_train.shape[1], activation='relu'),
                Dense(units=24, activation='relu'),
                Dropout(0.5),
                Dense(units=20, activation='relu'),
                Dense(units=24, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.summary()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            nb_epoch = 5
            batch_size = 32
            model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size)
            pred1 = model.predict(X_test)
            model.evaluate(X_test, y_test)
            accuracy2 = accuracy_score(y_test, pred1.round())
            precision2 = precision_score(y_test, pred1.round())
            recall2 = recall_score(y_test,pred1.round())
            # model.save('model_4.h5')
            return render_template('model.html',msg= 'accuracy',score = accuracy2,selected = 'NEURAL NETWORKS')

        elif model == 3:
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            from tensorflow.keras import regularizers
            df = df1.drop(['Time'], axis=1)
            X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
            X_train = X_train[X_train.Class == 0]
            X_train = X_train.drop(['Class'], axis=1)
            y_test = X_test['Class']
            X_test = X_test.drop(['Class'], axis=1)
            X_train = X_train.values
            X_test = X_test.values
            X_train.shape
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="tanh",
                            activity_regularizer=regularizers.l1(10e-5))(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
            decoder = Dense(input_dim, activation='relu')(decoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            nb_epoch = 5
            batch_size = 32
            autoencoder.compile(optimizer='adam',
                                loss='mean_squared_error',
                                metrics=['accuracy'])

            history = autoencoder.fit(X_train, X_train,
                                      epochs=nb_epoch,
                                      batch_size=batch_size,

                                      validation_data=(X_test, X_test)).history
            predictions = autoencoder.predict(X_test)
            mse = pd.np.mean(pd.np.power(X_test - predictions, 2), axis=1)
            error_df = pd.DataFrame({'reconstruction_error': mse,
                                     'true_class': y_test})
            threshold = 10
            y_pred3 = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
            conf_matrix = confusion_matrix(error_df.true_class, y_pred3)
            accuracy3 = accuracy_score(error_df.true_class, y_pred3)
            precision3 = precision_score(error_df.true_class, y_pred3)
            recall3 = recall_score(error_df.true_class, y_pred3)

            return render_template('model.html',msg = 'accuracy',score = accuracy3,selected = 'AUTO ENCODERS')

        elif model == 4:
            kmeans = KMeans(n_clusters=2, init='k-means++', )
            model1 = kmeans.fit(X)
            pre = model1.predict(X)
            accuracy4 = accuracy_score(y, pre)
            precision4 = precision_score(y, pre)
            recall4 = recall_score(y, pre)
            return render_template('model.html',msg = 'accuracy',score=accuracy4,selected = 'K-MEANS CLUSTERING')
    return render_template('model.html')

@app.route('/graph',methods = ["POST","GET"])
def graph():
    print('ihdweud')
    print('jhdbhsgd')
    line_chart = pygal.Bar()
    line_chart.x_labels= ['Logistic Regression','Neural Network','Auto Encoders','K-Means Clustering']
    print('jdjkfdf')
    line_chart.add('RECALL', [recall1,recall2,recall3,recall4])
    print('1')
    line_chart.add('PRECISION', [precision1,precision2,precision3,precision4])
    print('2')
    line_chart.add('ACCURACY', [accuracy1,accuracy2,accuracy3,accuracy4])
    print('3')
    graph_data = line_chart.render()
    print('4')
    return render_template('graphs.html', graph_data=graph_data)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method=='POST':
        tm = float(request.form['f1'])
        v1 = float(request.form['f2'])
        v2 = float(request.form['f3'])
        v3 = float(request.form['f4'])
        v4 = float(request.form['f5'])
        v5 = float(request.form['f6'])
        v6 = float(request.form['f7'])
        v7 = float(request.form['f8'])
        v8 = float(request.form['f9'])
        v9 = float(request.form['f10'])
        v10 = float(request.form['f11'])
        v11 = float(request.form['f12'])
        v12 = float(request.form['f13'])
        v13 = float(request.form['f14'])
        v14 = float(request.form['f15'])
        v15 = float(request.form['f16'])
        v16 = float(request.form['f17'])
        v17 = float(request.form['f18'])
        v18 = float(request.form['f19'])
        v19 = float(request.form['f20'])
        v20 = float(request.form['f21'])
        v21 = float(request.form['f22'])
        v22 = float(request.form['f23'])
        v23 = float(request.form['f24'])
        v24 = float(request.form['f25'])
        v25 = float(request.form['f26'])
        v26 = float(request.form['f27'])
        v27 = float(request.form['f28'])
        v28 = float(request.form['f29'])
        amt = float(request.form['f30'])

        l = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
           v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amt]
        print(l)

        model = load_model(r'models/model_final.h5')
        pred = round(model.predict([l])[0][0])
        print(pred)
        if pred==0:
            val='Normal'
        else:
            val='Fraud'
        return render_template('prediction.html', pred=val)
    return render_template('prediction.html')

if __name__ == ('__main__'):
    app.run(debug=True)

# a=[-4.397974442, 1.358367028, -2.592844218, 2.679786967, -1.128130942, -1.706536388, -3.496197293, -0.248777743, 0.247767899, -4.801637406, 4.895844223, -10.91281932, 0.184371686, -6.771096725, -0.007326183, -7.358083221, -12.59841854, -5.131548628, 0.308333946, -0.171607879, 0.573574068, 0.176967718, -0.436206884, -0.053501865, 0.252405262, -0.657487755, -0.827135715, 0.84957338, 59.0]
# b=[[0.6797161]]
