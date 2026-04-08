# test_env.py
# Simple test script to verify the environment works.

from env.robot_env import RobotEnvironment

def main():
    # Create environment with 1 robot and 50 dirt particles
    env = RobotEnvironment(num_bots=1, num_dirt=50)

    # Reset to get initial observation
    obs = env.reset()

    # Run for up to 500 steps
    for i in range(500):
        obs, reward, done, info = env.step(None)
        if i % 50 == 0:
            print(
                f"[{i:04d} Steps] 覆盖率: {info['coverage']:>6.2%} | 碰撞: {info['collisions']:>3} 次 | 剩余灰尘: {info['remaining_dirt']}")

        if done:
            print(f"\n✅ 任务触发停止条件 (Steps: {info['steps']})")
            break

    # Print final coverage
    final_results = env.get_metrics()
    print("\n" + "=" * 40)
    print("Performance evaluation ")
    print("=" * 40)
    for key, value in final_results.items():
        print(f"{key:20}: {value}")
    print("=" * 40)
    # Close the window
    env.close()

if __name__ == "__main__":
    main()