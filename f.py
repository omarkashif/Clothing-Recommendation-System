import streamlit as st
import pandas as pd
import numpy as np
import time
import random

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon=":shirt:",
    layout="wide",
    initial_sidebar_state="collapsed",
    # background_image="https://images.unsplash.com/photo-1521747116042-5a810fda9664",
)


class ClothingEnvironment:
    def __init__(self, pants, shirts):
        self.pants = pants
        self.shirts = shirts
        self.num_pants = len(pants)
        self.num_shirts = len(shirts)
        self.reset()

    def reset(self):
        self.current_pant_idx = np.random.randint(self.num_pants)
        self.current_pant_desc = self.pants[self.current_pant_idx]

        matching_shirts = [s for s in self.shirts if s in self.current_pant_desc]
        if matching_shirts:
            self.current_shirt_desc = np.random.choice(matching_shirts)
        else:
            self.current_shirt_desc = np.random.choice(self.shirts)

        self.reward = 0

    def step(self, action):
        """
        Action: index of the shirt description to choose as the output
        Returns:
        - observation: the next pant description, or a dummy observation if the episode is over
        - reward: the reward for the chosen action, or None if the episode is over
        - done: whether the episode is over
        """
        chosen_shirt_desc = self.shirts[action]
        if chosen_shirt_desc in self.current_pant_desc:
            self.reward = 1
        else:
            self.reward = -1

        self.current_pant_idx = np.random.randint(self.num_pants)
        self.current_pant_desc = self.pants[self.current_pant_idx]
        observation = self.current_pant_desc
        done = False
        if (
            np.random.rand() < 0.2
        ):  # Set the probability of terminating the episode to 0.2
            done = True
            observation = np.zeros_like(self.current_pant_desc)
            reward = None
        return observation, self.reward, done

    def render(self, chosen_shirt_desc):
        st.write(f"Current Pant Description: {self.current_pant_desc}")
        st.write(f"Chosen Shirt Description: {chosen_shirt_desc}")
        st.write(f"Reward: {self.reward}")
        time.sleep(1)


class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate=0.003,
        discount_factor=0.89,
        exploration_rate=2.0,
        exploration_decay_rate=0.99,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((env.num_pants, env.num_shirts))

    def choose_action(self, state):
        if np.random.uniform() < self.exploration_rate:
            # Take a random action
            action = np.random.randint(self.env.num_shirts)
        else:
            # Choose the action with the highest Q-value
            action = np.argmax(self.q_table[state, :])
        return action

    def update_q_table(self, state, action, reward, next_state):
        # Update the Q-value for the current state-action pair
        current_q = self.q_table[state, action]
        next_q = np.max(self.q_table[next_state, :])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        self.q_table[state, action] = new_q

    def run_episode(self):
        state = self.env.current_pant_idx
        done = False
        while not done:
            action = self.choose_action(state)
            observation, reward, done = self.env.step(action)
            next_state = self.env.current_pant_idx
            self.update_q_table(state, action, reward, next_state)
            state = next_state
        self.exploration_rate *= self.exploration_decay_rate


# Load the dataset
df = pd.read_csv("dataset.csv")
p = df.pant_description
s = df.shirt_description
p = list(p[:])
s = list(s[:])

# Initialize the environment
pants = p
shirts = s
env = ClothingEnvironment(pants, shirts)

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent by running multiple episodes
for i in range(1000):
    agent.run_episode()

# Create a Streamlit app
st.title("AdaptiveCloset: AI-Powered Clothing Recommender")


option1, option2, option3, option4 = np.random.choice(pants, 4, replace=False)
option_dict = {1: option1, 2: option2, 3: option3, 4: option4}

# Display the options as buttons
option_choice = 1
if st.button(f"1. {option1}"):
    option_choice = 1
elif st.button(f"2. {option2}"):
    option_choice = 2
elif st.button(f"3. {option3}"):
    option_choice = 3
elif st.button(f"4. {option4}"):
    option_choice = 4

# Make a recommendation based on the user input
if option_choice:
    cp = option_dict[option_choice]
    state = pants.index(cp)
    # st.write("Thinking...")
    # time.sleep(3)  # Add a delay of 3 seconds
    action = agent.choose_action(state)
    recommended_shirt = shirts[action]

    # Freeze the screen for 3 seconds
    # st.write("Displaying the recommended shirt...")
    # time.sleep(0.001)  # Add a delay of 3 seconds
    st.write(f"Recommended shirt: {recommended_shirt}")

    # Ask the user for feedback
    feedback = st.radio("Did you like the recommendation?", ("Yes", "No"))

    # Update the Q-table with the user's feedback
    if feedback == "Yes":
        reward = 1
    else:
        reward = -1
    next_state = env.current_pant_idx
    agent.update_q_table(state, action, reward, next_state)

    # Update the exploration rate
    agent.exploration_rate *= agent.exploration_decay_rate
    # time.sleep(2)  # Add a delay of 3 seconds


# Get user input for pant description
# option1, option2, option3, option4 = np.random.choice(pants, 4, replace=False)
# option_dict = {1: option1, 2: option2, 3: option3, 4: option4}
# option_choice = st.radio("Choose a pant description:",
#                          (f"1. {option1}", f"2. {option2}", f"3. {option3}", f"4. {option4}"))

# # Make a recommendation based on the user input
# if option_choice:
#     c = int(option_choice[0])
#     cp = option_dict[c]
#     state = pants.index(cp)
#     # st.write("Thinking...")
#     # time.sleep(3)  # Add a delay of 3 seconds
#     action = agent.choose_action(state)
#     recommended_shirt = shirts[action]

#     # Freeze the screen for 3 seconds
#     # st.write("Displaying the recommended shirt...")
#     time.sleep(0.001)  # Add a delay of 3 seconds
#     st.write(f"Recommended shirt: {recommended_shirt}")

#     # Ask the user for feedback
#     feedback = st.radio("Did you like the recommendation?", ("Yes", "No"))

#     # Update the Q-table with the user's feedback
#     if feedback == "Yes":
#         reward = 1
#     else:
#         reward = -1
#     next_state = env.current_pant_idx
#     agent.update_q_table(state, action, reward, next_state)

#     # Update the exploration rate
#     agent.exploration_rate *= agent.exploration_decay_rate
#     time.sleep(2)  # Add a delay of 3 seconds
