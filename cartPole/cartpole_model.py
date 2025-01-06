from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

def model():

    x_input = Input((4,))

    x = Dense(512, input_shape=4, activation='relu' )(x_input)

    #hidden layer wit 256 nodes
    x = Dense(256, activation='relu')(x)

    #hidden layer with 64 nodes
    x = Dense(64,  activation='relu')(x)

    #output layer
    x = Dense(2, activation='linear')(x)

    model = Model(inputs = x_input, outputs = x, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["mse"])

    model.summary()
    return model