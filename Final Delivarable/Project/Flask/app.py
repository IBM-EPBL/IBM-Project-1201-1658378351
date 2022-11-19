import pickle 
import cv2 
from flask import Flask, request, render_template 
import os.path
from skimage import feature
app=Flask(__name__,template_folder='./Templates',static_folder="./uploads")


@app.route("/")
def about():
    return render_template("about.html") 
@app.route("/about") 
def home(): 
    return render_template("about.html") 
@app.route("/info")
def information():
    return render_template("info.html")
@app.route("/upload")  
def test(): 
    return render_template("index.html") 
@app.route('/predict', methods=['GET', 'POST']) 
def upload(): 
    if request.method == 'POST':
        f=request.files [ 'myfile'] 
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath, "uploads", f.filename)
        f.save(filepath)
        print("[INFO] Loading model...")
        model = pickle.loads (open('parkinson.pkl', "rb").read())
        image= cv2.imread(filepath) 
        output =image.copy() 
        output= cv2.resize(output, (128, 128)) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image= cv2.resize(image, (200, 200)) 
        image=cv2.threshold (image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
        features = feature.hog(image, orientations=9,pixels_per_cell=(10, 10), cells_per_block=(2, 2),transform_sqrt=True, block_norm="L1") 
        preds=model.predict([features])
        print(preds)
        print(os.getcwd())
        ls=["Healthy", "Parkinson"]
        result = ls[1]
        if result == "Healthy" :
            col="green"
        else :
            col="red"
    return render_template("result.html",path="./uploads/"+f.filename,res=result,col=col)

if __name__=="__main__": 

    app.run(debug=True)

