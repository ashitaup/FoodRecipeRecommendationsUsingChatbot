import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools

downloads_path = f'/Users/ashitaupadhyay/Downloads'
saved_model_path = os.path.join(downloads_path, 'VStrained_model.h5')
loaded_model = load_model(saved_model_path)

validation_set= tf.keras.utils.image_dataset_from_directory(
    os.path.join(downloads_path, 'fruit-and-vegetable-image-recognition/validation'),
    labels = 'inferred',
    label_mode= 'categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224,224),
    shuffle= True,
    seed=None,
    validation_split=None,
    subset= None,
    interpolation = "bilinear",
    follow_links= False,
    crop_to_aspect_ratio = False

)
test_set = tf.keras.utils.image_dataset_from_directory(
    os.path.join(downloads_path, 'fruit-and-vegetable-image-recognition/test'),
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

training_set= tf.keras.utils.image_dataset_from_directory(
   os.path.join(downloads_path, 'fruit-and-vegetable-image-recognition/train'),
    labels = 'inferred',
    label_mode= 'categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224,224),
    shuffle= True,
    seed=None,
    validation_split=None,
    subset= None,
    interpolation = "bilinear",
    follow_links= False,
    crop_to_aspect_ratio = False

)

test_results = loaded_model.evaluate(test_set)
print("Test Accuracy:", test_results[1])
print("Test Loss:", test_results[0])
img_path = os.path.join(downloads_path, 'potato.jpeg')
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = loaded_model.predict(img_array)
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

print("Predicted Class:", predicted_class)

predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
predicted_class_name = training_set.class_names[predicted_class_index]
print("Predicted Class Index:", predicted_class_index)
print("Predicted Class Name:", predicted_class_name)

'''loaded_model = tf.keras.models.load_model('VStrained_model.h5')
true_labels = []
predicted_labels = []
for images, labels in validation_set:
    true_labels.extend(np.argmax(labels, axis=1))  # Get true labels
    predictions = loaded_model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))  # Get predicted labels

class_names = validation_set.class_names
report = classification_report(true_labels, predicted_labels, target_names=class_names)

print("Classification Report:\n", report)

confusion = confusion_matrix(true_labels, predicted_labels)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(confusion, class_names)
plt.show()
cnn= tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[224,224,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=36, activation='softmax'))

cnn.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn.summary())


def get_true_labels_and_predictions(model, dataset):
    true_labels = []
    predicted_labels = []
    for images, labels in dataset:
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
        predicted_labels.extend(np.argmax(model.predict(images), axis=1))
    return true_labels, predicted_labels

# Calculate metrics for the training dataset
train_true_labels, train_pred_labels = get_true_labels_and_predictions(cnn, training_set)
train_cm = confusion_matrix(train_true_labels, train_pred_labels)
train_report = classification_report(train_true_labels, train_pred_labels)

# Calculate metrics for the validation dataset
val_true_labels, val_pred_labels = get_true_labels_and_predictions(cnn, validation_set)
val_cm = confusion_matrix(val_true_labels, val_pred_labels)
val_report = classification_report(val_true_labels, val_pred_labels)

# Calculate metrics for the test dataset
test_true_labels, test_pred_labels = get_true_labels_and_predictions(cnn, test_set)
test_cm = confusion_matrix(test_true_labels, test_pred_labels)
test_report = classification_report(test_true_labels, test_pred_labels)

# Print or use the metrics as needed
print("Train Confusion Matrix:")
print(train_cm)
print("Train Classification Report:")
print(train_report)

print("\nValidation Confusion Matrix:")
print(val_cm)
print("Validation Classification Report:")
print(val_report)

print("\nTest Confusion Matrix:")
print(test_cm)
print("Test Classification Report:")
print(test_report)

def normalize_confusion_matrix(cm):
    # Normalize each row of the confusion matrix to sum up to 1
    row_sums = cm.sum(axis=1)
    normalized_cm = cm / row_sums[:, np.newaxis]
    return normalized_cm

# ... (your existing code for getting true labels and predictions and calculating metrics) ...

# Normalize confusion matrices
train_cm_normalized = normalize_confusion_matrix(train_cm)
val_cm_normalized = normalize_confusion_matrix(val_cm)
test_cm_normalized = normalize_confusion_matrix(test_cm)

# Function to plot the normalized confusion matrix
def plot_normalized_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
class_names = sorted(set(train_true_labels + val_true_labels + test_true_labels))

# Plot normalized confusion matrices for all datasets
plot_normalized_confusion_matrix(train_cm_normalized, class_names)
plot_normalized_confusion_matrix(val_cm_normalized, class_names)
plot_normalized_confusion_matrix(test_cm_normalized, class_names)

num_samples_to_display = 9
sample_images, sample_labels = next(iter(test_set.take(num_samples_to_display)))
class_names = test_set.class_names
sample_predictions = cnn.predict(sample_images)
sample_predictions = np.argmax(sample_predictions, axis=1)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i].numpy().astype(np.uint8))
    true_label = class_names[np.argmax(sample_labels[i])]
    predicted_label = class_names[sample_predictions[i]]
    ax.set_title(f"True: {true_label}\nPredicted: {predicted_label}")

plt.tight_layout()
plt.show()'''
