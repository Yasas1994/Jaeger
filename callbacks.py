
def scheduler(epoch, lr):
    if epoch <= 10:
        if epoch%2 ==0:

                return lr * tf.math.exp(-0.1)
        else:
            return lr
    else:
         return lr * tf.math.exp(-0.1)


#Keras callbacks
EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', patience=20, verbose=1,
    mode='max', restore_best_weights=True
)

ModelCheckPoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='val_loss', verbose=0,
    save_weights_only=True, mode='min', save_best_only=True
)
CSVLogger = tf.keras.callbacks.CSVLogger(
    filename=log , separator=',', append=True
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)

LRScheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
ReduceLr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=1, min_lr=1e-8)


