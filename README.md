# Hazrat Ali

# Software Engineering




# from matplotlib import style
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from matplotlib.gridspec import GridSpec
# from plotly.subplots import make_subplots
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)
# sns.set()
# #style.use('fivethirtyeight')
# pd.options.mode.chained_assignment = None

# import pandas as pd
# import numpy as np
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer

# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications import MobileNetV2








# Data Science Math Command : 
 
    df = pd.read_csv('dataset.csv')
    df.head()
    df.tail()
    df.info()
    df.describe()
    df.isnull().sum()
    df.columns
    df.hist(figsize=(20,15))
    plt.show()
    sns.pairplot(df.iloc[:,0:6])
    plt.show()
    df.drop(['name'],axis=1,inplace=True)
    df.corr()["status"][:-1].sort_values().plot(kind="bar")


    class_names = os.listdir(images)
    print(class_names)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    image_label = le.fit_transform(class_names)
    print(image_label)
     
    image_data = []
    image_labels = []

    for class_name in class_names:
    class_path = os.path.join(images, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)/255
        image_data.append(img_array)
        image_labels.append(class_name) 


        from sklearn.preprocessing import LabelEncoder
         le = LabelEncoder()
        image_labels = le.fit_transform(image_labels)
        print(image_labels)


    from sklearn.model_selection import train_test_split
    train_images, test_images, train_labels, test_labels = train_test_split(image_data, image_labels, test_size=0.1, random_state=42)    


    print(len(train_images))
    print(len(test_images))

    plt.figure(figsize=(10, 5))
    for i in range(20):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])

    plt.show()

     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
     base_model.trainable = False
     model = models.Sequential()
     model.add(base_model)
     model.add(layers.GlobalAveragePooling2D())
     model.add(layers.Dense(128, activation='relu'))
     model.add(layers.Dropout(0.2))
     model.add(layers.Dense(64, activation='relu'))
     model.add(layers.Dropout(0.2))
     model.add(layers.Dense(32, activation='relu'))
     model.add(layers.Dense(8, activation='softmax'))


     train_images = np.array(train_images)
     test_images = np.array(test_images)
     train_labels = np.array(train_labels)
     test_labels = np.array(test_labels)


     model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


     import numpy as np
     import matplotlib.pyplot as plt

# Make predictions for all test images
    predictions = model.predict(test_images)

# Get predicted class indices for all images
    predicted_indices = np.argmax(predictions, axis=1)

# Convert all predicted indices to class names
    predicted_class_names = le.inverse_transform(predicted_indices)

    def display_prediction(index): 
    plt.imshow(test_images[index], cmap=plt.cm.binary)
    plt.title(f"Predicted: {predicted_class_names[index]}")
    plt.axis('off')
    plt.show()

# Display first 5 predictions
    for i in range(5):
    display_prediction(i)    

    import tensorflow as tf
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing import image
 
    def load_and_preprocess_image(image_path, img_size):
    """Load, resize, and normalize an image."""
    img = image.load_img(image_path, target_size=img_size)  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize (if needed)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img, img_array  # Return both original and processed images   


              




    

