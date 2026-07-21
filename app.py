from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("linear_model.pkl", "rb") as file:
    model = pickle.load(file)

HTML = """
<!DOCTYPE html>
<html>
<head>

<title>Linear Regression Predictor</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial,Helvetica,sans-serif;
}

body{

background:linear-gradient(-45deg,#1e3c72,#2a5298,#6dd5ed,#2193b0);
background-size:400% 400%;
animation:bg 12s ease infinite;
height:100vh;
display:flex;
justify-content:center;
align-items:center;

}

@keyframes bg{

0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}

}

.container{

width:90%;
max-width:950px;

background:rgba(255,255,255,.15);

backdrop-filter:blur(18px);

padding:40px;

border-radius:20px;

box-shadow:0 10px 35px rgba(0,0,0,.3);

animation:slide 1s;

}

@keyframes slide{

from{
opacity:0;
transform:translateY(50px);
}

to{
opacity:1;
transform:translateY(0px);
}

}

h1{

text-align:center;
color:white;
margin-bottom:25px;

}

.grid{

display:grid;
grid-template-columns:repeat(4,1fr);
gap:15px;

}

input{

padding:12px;
border:none;
border-radius:10px;
outline:none;
font-size:15px;
transition:.3s;

}

input:focus{

transform:scale(1.05);
box-shadow:0 0 12px cyan;

}

button{

margin-top:25px;
width:100%;
padding:15px;
font-size:18px;
font-weight:bold;

border:none;

border-radius:12px;

background:#00e5ff;

cursor:pointer;

transition:.4s;

}

button:hover{

background:#00bcd4;
transform:scale(1.03);

}

.result{

margin-top:25px;

padding:20px;

background:white;

border-radius:12px;

text-align:center;

font-size:25px;

font-weight:bold;

color:#1565c0;

animation:pop .5s;

}

@keyframes pop{

0%{transform:scale(.5);}
100%{transform:scale(1);}

}

.footer{

text-align:center;

color:white;

margin-top:20px;

font-size:14px;

}

@media(max-width:900px){

.grid{

grid-template-columns:repeat(2,1fr);

}

}

@media(max-width:550px){

.grid{

grid-template-columns:1fr;

}

}

</style>

</head>

<body>

<div class="container">

<h1>📈 Linear Regression Prediction</h1>

<form method="POST">

<div class="grid">

{% for i in range(16) %}

<input type="number" step="any" name="f{{i}}" placeholder="Feature {{i+1}}" required>

{% endfor %}

</div>

<button type="submit">Predict</button>

</form>

{% if prediction %}

<div class="result">

Prediction : {{prediction}}

</div>

{% endif %}

<div class="footer">

Designed using Flask • Render • Machine Learning

</div>

</div>

</body>

</html>

"""

@app.route("/", methods=["GET","POST"])

def home():

    prediction=None

    if request.method=="POST":

        try:

            features=[float(request.form[f"f{i}"]) for i in range(16)]

            final=np.array(features).reshape(1,-1)

            pred=model.predict(final)[0]

            prediction=round(float(pred),3)

        except Exception as e:

            prediction=str(e)

    return render_template_string(HTML,prediction=prediction)

if __name__=="__main__":

    app.run(debug=True)
