import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print((train_images.shape))
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
model.load_weights('model.h5')

'''
show_imgs(test_images[:16])

predictions = np.argmax(model.predict(test_images[:16]), 1)
print(class_names[x] for x in predictions)
'''

predictions = model.predict(test_images)
Y = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = Y.shape[1]
print(n_classes)
print(Y.shape)

'''
print("\nPredicted matrix: ", predictions[0])

print("\nPredicted value: ", np.argmax(predictions[0]))

print("\nTest label:", test_labels[0])

predictions = np.argmax(predictions, axis=1)
'''

# Mean Average Precision
average_precision = dict()

for i in range(n_classes):
    average_precision[i] = average_precision_score(Y[:, i], predictions[:, i])

average_precision["micro"] = average_precision_score(Y, predictions, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

for i in range(25):
    k = np.argmax(predictions[i])
    print(class_names[k])

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel(class_names[test_labels[i][0]])
plt.show()


