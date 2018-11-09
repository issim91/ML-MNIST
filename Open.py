# coding: utf-8

from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils


# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_test = X_test.reshape(10000, 784)

# Нормализация данных
X_test = X_test.astype('float32')
X_test /= 255

# Преобразуем метки в категории
Y_test = np_utils.to_categorical(y_test, 10)

# Загружаем данные об архитектуре сети из файла json
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)

# Загружаем веса в модель
loaded_model.load_weights("mnist_model.h5")

# Компилируем модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Проверяем модель на тестовых данных
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Точность модели на тестовых данных: %.2f%%" % (scores[1]*100))