import tensorflow as tf
from tensorflow.keras import Model


class SingleNet(tf.keras.Model):
    def __init__(
            self, 
            encoder,
            u_net,
        ):
        super(SingleNet, self).__init__()

        self.model_predict = Model(
            inputs=encoder.inputs,
            outputs=[
                u_net(
                    encoder(
                        encoder.inputs
                    )
                )[0]
            ]
        )

    def compile(
        self,
        optimizer,
        metrics,
        loss_fn,
    ):
        """ Configure the model.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(SingleNet, self).compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            predictions = self.model_predict(x, training=True)
            loss = self.loss_fn(y, predictions)

        model_gradients = tape.gradient(loss, self.model_predict.trainable_variables)
        self.optimizer.apply_gradients(zip(model_gradients, self.model_predict.trainable_variables))
        self.compiled_metrics.update_state(y, predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {
                "source_loss": loss,
            }
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.model_predict(x, training=False)

        # Calculate the loss
        loss = self.loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"source_loss": loss})
        return results
    
    def call(self, inputs):
        return self.model_predict(inputs)


class DannNet(tf.keras.Model):
    def __init__(
            self, 
            source_encoder, 
            target_encoder, 
            u_net, 
            domain_classifyer,
        ):
        super(DannNet, self).__init__()

        self.s_model_predict = Model(
            inputs=source_encoder.inputs,
            outputs=[
                u_net(
                    source_encoder(
                        source_encoder.inputs
                    )
                )[0]
            ]
        )
        self.s_model_domain = Model(
            inputs=source_encoder.inputs,
            outputs=[
                domain_classifyer(
                    u_net(
                        source_encoder(
                            source_encoder.inputs
                        )
                    )[1]
                )
            ]
        )
        
        self.t_model_predict = Model(
            inputs=target_encoder.inputs,
            outputs=[
                u_net(
                    target_encoder(
                        target_encoder.inputs
                    )
                )[0]
            ]
        )
        self.t_model_domain = Model(
            inputs=target_encoder.inputs,
            outputs=[
                domain_classifyer(
                    u_net(
                        target_encoder(
                            target_encoder.inputs
                        )
                    )[1]
                )
            ]
        )
        

    def compile(
        self,
        optimizer,
        metrics,
        s_loss,
        t_loss
    ):
        """ Configure the model.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(DannNet, self).compile(optimizer=optimizer, metrics=metrics)
        self.s_loss_fn = s_loss
        self.t_loss_fn = t_loss
        self.d_loss_fn = tf.losses.BinaryCrossentropy()

    def train_step(self, data):
        # Unpack data
        x, y = data
        s_x, t_x = x 
        s_y, t_y, d_s_y, d_t_y = y
        
        # Forward
        with tf.GradientTape() as tape:
            s_predictions = self.s_model_predict(s_x, training=True)
            s_loss = self.s_loss_fn(s_y, s_predictions)
        s_model_gradients = tape.gradient(s_loss, self.s_model_predict.trainable_variables)

        with tf.GradientTape() as tape:
            t_predictions = self.t_model_predict(t_x, training=True)
            t_loss = self.t_loss_fn(t_y, t_predictions)
        t_model_gradients = tape.gradient(t_loss, self.t_model_predict.trainable_variables)

        with tf.GradientTape() as tape:
            d_s_predictions = self.s_model_domain(s_x, training=True)
            d_s_loss = self.d_loss_fn(d_s_y, d_s_predictions)
        d_s_model_gradients = tape.gradient(d_s_loss, self.s_model_domain.trainable_variables)

        with tf.GradientTape() as tape:
            d_t_predictions = self.t_model_domain(t_x, training=True)
            d_t_loss = self.d_loss_fn(d_t_y, d_t_predictions)
        d_t_model_gradients = tape.gradient(d_t_loss, self.s_model_domain.trainable_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(s_model_gradients, self.s_model_predict.trainable_variables))
        self.optimizer.apply_gradients(zip(t_model_gradients, self.t_model_predict.trainable_variables))
        self.optimizer.apply_gradients(zip(d_s_model_gradients, self.s_model_domain.trainable_variables))
        self.optimizer.apply_gradients(zip(d_t_model_gradients, self.t_model_domain.trainable_variables))

        # Update the metrics configured in `compile()`.
        t_y = tf.convert_to_tensor(t_y)
        self.compiled_metrics.update_state(t_y, t_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {
                "source_loss": s_loss, 
                "target_loss": t_loss,
                "domain_classifyer_loss": d_t_loss
            }
        )
        return results
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        s_x, t_x = x 
        s_y, t_y, d_s_y, d_t_y = y

        # Compute predictions
        y_prediction = self.t_model_predict(t_x, training=False)

        # Calculate the loss
        t_loss = self.t_loss_fn(t_y, y_prediction)

        # Update the metrics.
        t_y = tf.convert_to_tensor(t_y)
        self.compiled_metrics.update_state(t_y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"target_loss": t_loss})
        return results
    
    def call(self, inputs):
        s_x, t_x = inputs
        s_y = self.s_model_predict(s_x, training=False)
        t_y = self.t_model_predict(t_x, training=False)
        d_s_y = self.s_model_domain(s_x, training=False)
        d_t_y = self.t_model_domain(t_x, training=False)
        return s_y, t_y, d_s_y, d_t_y