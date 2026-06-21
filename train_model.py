import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical",
    subset="training"
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=8,
    class_mode="categorical",
    subset="validation"
)

print("Class Indices:")
print(train_generator.class_indices)

# Load MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(6, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Save model
model.save("food_model.h5")

print("Model saved successfully!")