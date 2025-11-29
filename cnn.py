import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 2. 전처리
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 3. CNN 모델 정의
model = models.Sequential([
   layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
   layers.MaxPooling2D((2, 2)),
   layers.Conv2D(64, (3, 3), activation='relu'),
   layers.MaxPooling2D((2, 2)),
   layers.Flatten(),
   layers.Dense(64, activation='relu'),
   layers.Dense(10, activation='softmax')
])


# 4. 학습 설정
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# 5. 학습
history = model.fit(x_train, y_train,
                   epochs=10,
                   batch_size=128,
                   validation_data=(x_test, y_test),
                   verbose=1)


# 6. 평가
train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# 7. LOSS 커브 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('CNN Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
