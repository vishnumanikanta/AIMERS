import logging
import cv2
import numpy as np 
import tensorflow as tf
import tempfile
import requests
from io import BytesIO
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import phonenumbers
from phonenumbers import geocoder, carrier, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from datetime import datetime
import io
import random
import pyjokes  # Added pyjokes library for joke generation

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
TOKEN = 'your telegram token'

# Load a pre-trained model, e.g., MobileNetV2 trained on ImageNet
mobilenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Class names for CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Define an improved CNN model for CIFAR-10
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

cifar_model = create_model()

referral_relationships = {}

def start(update: Update, context: CallbackContext):
    keyboard = [
        ['Help', 'Train Model'],
        ['Process Video'],
        ['Phone Number Info'],
        ['Date', 'Refer and Earn'],
        ['Joke']  # Added 'Joke' option to the menu
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    update.message.reply_text("Welcome! Choose an option:", reply_markup=reply_markup)

def help_command(update: Update, context: CallbackContext):
    update.message.reply_text("""
    /start - Starts the conversation
    /help - Shows this message
    /train - Trains the neural network
    /process_video - Processes a video and recognizes objects
    /phone_number_info - Get info about a phone number
    /joke - Get a random joke
    """)

def train(update: Update, context: CallbackContext):
    update.message.reply_text("Model training started. This may take a while... (3 minutes time padtundi agaandi kasepu,10 epoch iyaka pic petandi )")

    # Compile and train the CIFAR-10 model
    cifar_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('cifar_classifier.keras', save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    progress_callback = ProgressCallback(update, epochs=50)

    history = cifar_model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_cb, early_stopping_cb, progress_callback]
    )

    plot_training_history(history, update)

    update.message.reply_text("Training complete! You can now send a photo for prediction (training iyindi, photo pettu).")


class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, update, epochs):  # Add the epochs parameter
        super().__init__()
        self.update = update
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        message = f"Epoch {epoch + 1}/{self.epochs}: accuracy={accuracy:.4f}, val_accuracy={val_accuracy:.4f}, loss={loss:.4f}, val_loss={val_loss:.4f}"
        self.update.message.reply_text(message)


def plot_training_history(history, update):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    update.message.reply_photo(photo=buf)

def process_video(update: Update, context: CallbackContext):
    update.message.reply_text("Please send a video for processing.")

def handle_video(update: Update, context: CallbackContext):
    try:
        video_file = update.message.video.get_file()
        video_path = download_video(video_file.file_path)
        results = detect_objects_in_video(video_path)

        if results:
            for frame_result in results:
                update.message.reply_text(frame_result)
        else:
            update.message.reply_text("No objects detected in the video.")
    except Exception as e:
        logger.error(f"Error handling video: {e}")
        update.message.reply_text("An error occurred while processing the video. Please try again later.")


def download_video(file_path):
    response = requests.get(file_path)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

def detect_objects_in_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = mobilenet_model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return ", ".join([f"{p[1]}: {p[2] * 100:.2f}%" for p in decoded_preds])

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:
            result = detect_objects_in_frame(frame)
            frame_results.append(f"Frame {frame_count}: {result}")

        frame_count += 1

    cap.release()
    return frame_results

def send_joke(update: Update, context: CallbackContext):
    joke = pyjokes.get_joke()
    update.message.reply_text(joke)

def generate_referral_code():
    referral_code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
    print("Generated referral code:", referral_code)  # Print the generated code for verification
    return referral_code


def handle_phone_number(update: Update, context: CallbackContext):
    text = update.message.text.strip()
    try:
        # Parse the phone number
        parsed_number = phonenumbers.parse(text)

        # Check if the number is valid
        valid_number = phonenumbers.is_valid_number(parsed_number)

        # Get location information
        location = geocoder.description_for_number(parsed_number, "en")

        # Get carrier information
        carrier_name = carrier.name_for_number(parsed_number, "en")

        # Get timezone information
        timezones = timezone.time_zones_for_number(parsed_number)

        # Format the timezone information
        timezone_info = ", ".join(timezones) if timezones else "Unknown"

        # Prepare the response message
        response_message = f"Phone Number: {text}\n"
        response_message += f"Valid: {'Yes' if valid_number else 'No'}\n"
        response_message += f"Location: {location}\n"
        response_message += f"Carrier: {carrier_name}\n"
        response_message += f"Timezone: {timezone_info}"

        update.message.reply_text(response_message)
    except phonenumbers.NumberParseException:
        update.message.reply_text(
            "Invalid phone number. Please send a valid phone number in the format +[country code][number].")


def handle_image_filter(update: Update, context: CallbackContext):
    photo_file = update.message.photo[-1].get_file()
    photo_bytes = requests.get(photo_file.file_path).content
    img = cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = mobilenet_model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]
    result_text = ", ".join([f"{p[1]}: {p[2] * 100:.2f}%" for p in decoded_preds])
    update.message.reply_text(result_text)

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.strip().lower()
    if text == 'date':
        current_date = datetime.now().strftime("%Y-%m-%d")
        update.message.reply_text(f"The current date is: {current_date}")
    elif text == 'refer and earn':
        referral_code = generate_referral_code()
        referral_relationships[update.message.from_user.id] = referral_code
        update.message.reply_text(f"Your referral code is: {referral_code}. Share it with your friends!")
    elif text in ['hi', 'hello']:
        update.message.reply_text("Hi there! How can I assist you today?(nenu miku ala sahayapadagalanu)")
    elif text == 'train model':
        train(update, context)
    elif text == 'phone number info':
        update.message.reply_text("Please send a valid phone number in the format +[country code][number].")
    elif text == 'process video':
        update.message.reply_text("Please send a video for processing.")
    elif text == 'joke':
        send_joke(update, context)
    else:
        update.message.reply_text("Please choose a valid option from the menu.(menu nunchi oka option nokandi )")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(CommandHandler("process_video", process_video))
    dp.add_handler(CommandHandler("joke", send_joke))
    dp.add_handler(CommandHandler("phone_number_info", handle_phone_number))
    dp.add_handler(MessageHandler(Filters.text & Filters.regex(r'^\+'), handle_phone_number))  # For phone numbers

    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_image_filter))  # For photo messages
    dp.add_handler(MessageHandler(Filters.video, handle_video))  # For video messages
    dp.add_handler(MessageHandler(Filters.text & Filters.regex(r'^\+'), handle_phone_number))  # For phone numbers

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == "__main__":
    main()
