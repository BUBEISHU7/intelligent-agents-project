# test_env.py
# Simple test script to verify the environment works.

from env.robot_env import RobotEnvironment

def main():
    # Create environment with 1 robot and 50 dirt particles
    env = RobotEnvironment(num_bots=1, num_dirt=50)

    # Reset to get initial observation
    obs = env.reset()

    # Run for up to 500 steps
    for _ in range(500):
        obs, reward, done, info = env.step(None)
        if done:
            break

    # Print final coverage
    print(f"Coverage achieved: {info['coverage']:.2f}")

    # Close the window
    env.close()

if __name__ == "__main__":
    main()