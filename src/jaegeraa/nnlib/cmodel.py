import tensorflow as tf

    
class JaegerModel(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        
       
    def compile(self,loss_fn,optimizer,metrics,num_classes,**kwargs):
        super().compile()

        #tf.nn.softmax_cross_entropy_with_logits
        # Prepare the metrics.
        self.num_classes = num_classes
        self.step = 0
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics_ = metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regularization_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        
    
    def train_step(self, data):
        
        if len(data) == 3:
            #sample weights is class weights when a dictionary of class weights is provided to .fit
            x, y, sample_weights = data
            #tf.print(sample_weights)
        else:
            sample_weights = None
            x, y = data
            
        #tf.print(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            #y = tf.cast(y, dtype=y_logits.dtype) # cast to datatype of logits
            #tf.print(sample_weights)
            loss_scaled = self.loss_fn(y, y_pred['output'],sample_weights)
            loss_scaled += sum(self.losses)
            self.loss_tracker.update_state(loss_scaled)
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                loss_scaled = self.optimizer.get_scaled_loss(loss_scaled)
   

        
        grad = tape.gradient(loss_scaled, self.trainable_variables)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            grad = self.optimizer.get_unscaled_gradients(grad)

        self.optimizer.apply_gradients(zip(grad,self.trainable_variables))  # Loss scale is updated here
        
    # scaled_loss = opt.get_scaled_loss(loss)
    # scaled_grads = tape.gradient(scaled_loss, vars)
    # grads = opt.get_unscaled_gradients(scaled_grads)
    # opt.apply_gradients([(grads, var)])
    
        # Compute gradients
        
        #gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        #self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        #custom step tracker
        self.step +=1
        if self.step % 100 == 0:
            self.loss_tracker.reset_state()
        
        self.regularization_loss_tracker.update_state(sum(self.losses))

        return {"loss": self.loss_tracker.result(),
                "reg-loss": self.regularization_loss_tracker.result(),
                "lr":self.optimizer.learning_rate}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data

        y_pred = self(x, training=False)

        y = tf.cast(y, dtype= y_pred['output'].dtype)
        
        # Updates the metrics tracking the loss
        loss = self.loss_fn(y, y_pred['output'])
        self.loss_tracker.update_state(loss)
        
        for m in self.metrics:
            if 'loss' not in m.name:
                m.update_state(y, y_pred['output'])

        
        return {"loss": self.loss_tracker.result(),
                **{m.name:m.result() for m in self.metrics}
                }
    def predict_step(self, data):
        # Unpack the data
        x,y = data[0],data[1:]

        y_logits = self(data[0], training=False) #set model to inference mode
        return {'y_hat':y_logits,'meta':y}

    @property
    def metrics(self):
        out = []
        if hasattr(self, 'loss_tracker'):
            out.append(self.loss_tracker)
        if hasattr(self, 'attr_name'):
            out.extend(*self.metrics)
        return out
