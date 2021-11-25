from network import flamenco_net, data_generator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import math

BATCH_SIZE = 4

model = flamenco_net()

train_files = pickle.load(open('train_files.p', 'rb'))
val_files = pickle.load(open('val_files.p', 'rb'))

num_train_steps = int(math.floor(len(train_files) / BATCH_SIZE))
num_val_steps = int(math.floor(len(val_files) / BATCH_SIZE))

train_generator = data_generator(train_files,
                                 BATCH_SIZE,
                                 num_train_steps,
                                 True)

val_generator = data_generator(val_files,
                               BATCH_SIZE,
                               num_val_steps,
                               False)

callbacks = [EarlyStopping(patience=5),
             ModelCheckpoint('palo_tag.h5', save_best_only=True)]

model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch=num_train_steps,
                    epochs=100,
                    validation_data=val_generator,
                    validation_steps=num_val_steps,
                    callbacks=callbacks)


