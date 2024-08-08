import pathlib
from quadrotor_env import QuadrotorEnv
from learning_mpc import LearningMPC
from mujoco_mpc import agent as agent_lib

# Model path
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/quadrotor/task.xml"
)

# Initialize environment
env = QuadrotorEnv(model_path)

# Initialize MJPC agent
mjpc_agent = agent_lib.Agent(task_id="quadrotor", model=env.model)
mjpc_agent.set_cost_weights({"Velocity": 0.15})
mjpc_agent.set_task_parameter("Goal", -1.0)

# Initialize learning-based MPC controller
learning_mpc = LearningMPC(env.model)

# Run simulation and compare results
for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action_mjpc = mjpc_agent.act(state)
        action_learning_mpc = learning_mpc.predict(state)
        
        # Compare actions and results
        next_state, reward, done, _ = env.step(action_learning_mpc)
        learning_mpc.update(state, action_learning_mpc, reward, next_state)
        state = next_state

    learning_mpc.train(episodes=1)

# Evaluate and compare performance