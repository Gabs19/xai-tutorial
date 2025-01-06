import gym
import numpy as np
import shap
import tensorflow.keras.models as load_model


model_path = "database/cartpole-dqn.h5"
model = load_model(model_path)

env = gym.make('CartPole-v1')
state_samples = np.array([env.reset() for _ in range(100)])

explainer = shap.Explainer(model.predict, state_samples)
state_to_explain = env.reset().reshape(1, -1)
shap_values = explainer.shap_values(state_to_explain)

shap.summary_plot(shap_values, state_to_explain, feature_names=['Posição', 'Velocidade', "Ângulo", "Velocidade Angular"])

action = np.argmax(model.predict(state_to_explain))
print(f'Ação escolhida: {action}')
shap.dependence_plot(explainer.expected_value[0], shap_values[0], feature_names=['Posição', 'Velocidade', "Ângulo", "Velocidade Angular"])