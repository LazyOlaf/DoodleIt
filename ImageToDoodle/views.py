import io
import os
import json
import time
import ImageToDoodle.test as test

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.conf import settings

model = models.densenet121(pretrained=True)
model.eval()



def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


def transform_image(image_bytes):
    """
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    #tensor = transform_image(image_bytes)
    test.main(25,image_bytes)

    time.sleep(3.5)

    return "Rafi"

import base64
from django.shortcuts import render
from .forms import ImageUploadForm

def index(request):
    image_uri = None
    predicted_label = None
    img_height=256
    img_width=256
    if request.method == 'POST':
        # in case of POST: get the uploaded image from the form and process it
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # retrieve the uploaded image and convert it to bytes (for PyTorch)
            image = form.cleaned_data['image']

            image_bytes = image.file.read()
            new_image = Image.open(io.BytesIO(image_bytes))
            img_height = new_image.height
            img_width = new_image.width
            new_image.save('static/images/input.jpg',"JPEG")
            # convert and pass the image as base64 string to avoid storing it to DB or filesystem
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)




            # get predicted label with previously implemented PyTorch function
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)

    else:
        # in case of GET: simply show the empty form for uploading images
        form = ImageUploadForm()


    output_img = Image.open('static/images/input.jpg').resize((int(img_width/1),int(img_height/1)))
    output_img.save('static/images/input.jpg',"JPEG")
    print(output_img)
    time.sleep(0.5)

    # pass the form, image URI, and predicted label to the template to be rendered
    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': "",
    }
    return render(request, 'index.html', context)