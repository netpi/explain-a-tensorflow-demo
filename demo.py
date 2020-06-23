import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
data = mnist.load_data()
(x_train, y_train),(x_test, y_test) =  mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# def plot_image(image):
#     fig = plt.gcf() 
#     fig.set_size_inches(3,3)
#     plt.imshow(image, cmap='binary') 
#     plt.show() 

# Tensor = [
# [ [0,0,0], [255,255,255], [255,255,255] ],
# [ [255,255,255], [0,0,0], [255,255,255] ],
# [ [255,255,255], [255,255,255], [0,0,0] ]]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[y_train[i]])
plt.show()


# model.fit(x_train, y_train, epochs=10)
# model.evaluate(x_test, y_test)
