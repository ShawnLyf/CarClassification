import tensorflow as tf
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator() 
train_generator = datagen.flow_from_directory('data')

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply the same preprocessing function used during training
    img_array /= 255.0  # Scale the image like you did during training
    img_array = np.expand_dims(img_array, axis=0)  # Make the image array 4D: (1, 224, 224, 3)
    return img_array
def predict_image(model, img_path):
    img_array = load_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0], axis=-1)
    return predicted_class

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name
model = tf.keras.models.load_model('model.h5')
def index(request):
    message = ""
    fss = CustomFileSystemStorage()
    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        image_url = fss.url(_image)

        # load model
        
        # print("path")
        # print(path)
        predicted_class = predict_image(model, path)
        # print("predicted_class")
        # print(predicted_class)
        class_names = list(train_generator.class_indices.keys())
        # print("class_names")
        # print(class_names)
        
        carmodel=class_names[predicted_class]
        # print("carmodel")
        # print(carmodel)
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": carmodel,
            },
        )
    except MultiValueDictKeyError:

        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )
