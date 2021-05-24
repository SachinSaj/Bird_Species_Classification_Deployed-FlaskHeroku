#!/usr/bin/env python
# coding: utf-8

# In[50]:


from flask import Flask
from flask import Flask, render_template, request, redirect, url_for
from keras import models
import numpy as np
import cv2
import os
import string
import random
from PIL import Image
from werkzeug.utils import secure_filename


# In[51]:


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'static/uploads'

model = models.load_model('static/model/bird_species.h5')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        full_filename =  'images/white_bg.jpg'
        return render_template('index.html', full_filename = full_filename)
    
    if request.method == "POST":
        letters = string.ascii_lowercase
        name = ''.join(random.choice(letters) for i in range(10)) + '.png'
        full_filename =  'uploads/' + name
        
        image_upload = request.files['image_upload']
        if image_upload.filename != '':
            imagename = image_upload.filename
            image = Image.open(image_upload)
            image = image.resize((224,224))
            image.save(os.path.join(app.config['UPLOAD_PATH'], name))
            image_arr = np.array(image.convert('RGB'))
            image_arr.shape = (1,224,224,3)
            result = model.predict(image_arr)
            ind = np.argmax(result)
            classes = ['AMERICAN GOLDFINCH', 'BARN OWL', 'CARMINE BEE-EATER', 'DOWNY WOODPECKER', 'EMPEROR PENGUIN', 'FLAMINGO']
            return render_template('index.html', pred = classes[ind], full_filename = full_filename)


# In[52]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:




