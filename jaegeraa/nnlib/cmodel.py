import tensorflow as tf

class JaegerModel(tf.keras.Model):
    '''Custom model class with modified trainig loop
    '''
    
    def compile(self, optimizer):
        super(JaegerModel, self).compile()
        self.optimizer = optimizer
        
        # Prepare the metrics.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.auc_tracker = tf.keras.metrics.AUC(from_logits=True, name='auc')

    def train_step(self, data):

        x, y = data

        with tf.GradientTape() as tape:
            y_logits = self(x, training=True)  # Forward pass, set model to training mode

            # cross-entropy loss calculation
            loss=tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(y, dtype=y_logits.dtype), logits= tf.squeeze(y_logits), name='train-loss')
            loss=tf.reduce_mean(loss)
            loss += sum(self.losses) #add regularization losses

        # Compute gradients and update for all trainable variables
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        self.auc_tracker.update_state(y, y_logits)
        self.loss_tracker.update_state(loss)


        return {"loss": self.loss_tracker.result(), 
                "auc": self.auc_tracker.result(),
                "lr":self.optimizer.learning_rate}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data

        y_logits = self(x, training=False) #set model to inference mode
        # Compute predictions for reverse-complement
        #y_rclogits = self(x[::,::-1,::-1], training=False)
        
        # Updates the metrics tracking the loss
        losses=tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(y, dtype=y_logits.dtype), logits= tf.squeeze(y_logits), name='train-loss')
        loss = tf.reduce_mean(losses)

        self.loss_tracker.update_state(loss)

        # Update the metrics.
        self.auc_tracker.update_state(y, y_logits)


        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": self.loss_tracker.result(),
                "auc": self.auc_tracker.result()
               }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.auc_tracker]