from keras import layers
import keras
import keras.backend as K

class Attention():
    def __init__(self, rnn_units, batch_size, time_step, input_size, output_size):
        super(Attention, self).__init__()
        self.units = rnn_units
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_step = input_size
        self.output_size = output_size
        
        # model=tf.keras.models.Sequential()
        # model.add(layers.LSTM(self.units, return_sequences = True, return_state = True, stateful= True, recurrent_initializer = 'glorot_uniform'))
        # model.add(layers.Dense(self.units))
        # model.add(layers.Dense(self.units))
        # model.add(layers.Dense(1))
        # model.add(layers.Dense(1))
        # model.add(layers.Dense(self.output_size))
    def model(self,train):
        Input=layers.Input(shape=(self.time_step,1))

        lstm = layers.Bidirectional(layers.LSTM(self.units,return_sequences = True,recurrent_initializer='truncated_normal'))(Input) 
        
        attention=layers.TimeDistributed(layers.Dense(1,activation='tanh'))(lstm)
        attention=layers.Flatten()(attention)
        attention=layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(self.units*2)(attention)
        attention = layers.Permute([2,1])(attention)

        #lstm = layers.Bidirectional(layers.LSTM(self.units,return_sequences = True,recurrent_initializer='truncated_normal'))(attention) 
        sent_representation=layers.Multiply()([lstm,attention])
        sent_representation=layers.Lambda(lambda xin: K.sum(xin,axis=-2),output_shape=(self.units*2,))(sent_representation)
        
        Wh = layers.Dense(self.units)(sent_representation)
        # self.Ws = tf.keras.layers.Dense(self.units)
        # self.Wx = tf.keras.layers.Dense(1)
        # self.V = tf.keras.layers.Dense(1)
        O = layers.Dense(self.output_size)(Wh)

        model=keras.models.Model(inputs=Input,outputs=O)
        
        if train==False:
            model = keras.models.load_model('model/model.h5')
        
        optimizer=keras.optimizers.Adam(0.0001)
        model.compile(loss='mse',optimizer=optimizer)

        return model