import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
from datasets import load_from_disk
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
# load data
main_data = load_from_disk("data/en-jp-v3.0-subset-sc-ov-77")

# define hyperparameters
hyperparams = {"max_length": 256,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay_rate": 0.0,
                "epochs": 6}

# load tokenizer, model,  and data_collator
checkpoint  = "Helsinki-NLP/opus-mt-en-jap"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt", private=True, access_token=access_token)
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

# function to batch tokenize the data
def process_fn(batch):
  inputs = [x["en"] for x in batch["translation"]]
  targets = [x["jp"] for x in batch["translation"]]
  return tokenizer(inputs, text_target=targets, max_length=hyperparams["max_length"], truncation=True)

# batch tokenize the data
tokenized_datasets = main_data.map(
    process_fn,
    batched=True,
    remove_columns=main_data["train"].column_names
) 

# convert train and validation data to tf.data.dataset
train_data_tf = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=hyperparams["batch_size"]
)

validation_data_tf = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16
)

# specify num of training step
num_train_steps = len(train_data_tf) * hyperparams["epochs"]

# define linear warmup cosine decay scheduler
class LinearWarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, num_warmup_steps, num_training_steps):
        super(LinearWarmupCosineSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, step_num):
        warmup_completed = tf.cast(step_num >= self.num_warmup_steps, tf.float32)
        warmup_progress = tf.cast(step_num, tf.float32) / tf.cast(self.num_warmup_steps, tf.float32)
        training_progress = (tf.cast(step_num, tf.float32) - tf.cast(self.num_warmup_steps, tf.float32)) / tf.cast(self.num_training_steps - self.num_warmup_steps, tf.float32)

        cosine_decay = 0.5 * (1.0 + tf.math.cos(tf.constant(np.pi) * training_progress))
        return self.initial_learning_rate * (1.0 - warmup_completed) * warmup_progress + self.initial_learning_rate * warmup_completed * cosine_decay

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
        }

# specify warmup steps, scheduler, and optimizer
warmup_steps = int(0.1 * num_train_steps)
schedule = LinearWarmupCosineSchedule(hyperparams["learning_rate"], warmup_steps, num_train_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule, weight_decay=hyperparams["weight_decay_rate"])

# compile model with default cross entropy loss
model.compile(optimizer=optimizer)

# to make training memory efficient
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Define callbacks
tensorboard_callback = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    update_freq='epoch')

hub_callback = PushToHubCallback(
    output_dir="ja-en-dataset-v3.0-subset-v3.0",
    tokenizer=tokenizer
)

# fit model, define any callbacks if you want to use one
model.fit(
    train_data_tf,
    validation_data=validation_data_tf,
    callbacks=[tensorboard_callback, hub_callback],
    epochs=hyperparams["epochs"]
)