import gym

action_mapping = {
    'l': 0,
    'd': 1,
    'r': 2,
    'u': 3
}


def get_input():
    action = None
    while True:
        x = input("Provide an input (u)p, (d)own, (l)eft, r(ight): \t")
        x = x.lower()
        if x in action_mapping:
            action = action_mapping[x]
            break
        else:
            print("Wrong input.")
    return action

def interact(env):
    env.reset()
    env.render()
    done = False
    cum_reward = 0
    while not done:
        action = get_input()
        obs, rew, done, info = env.step(action)
        print(f"Env return: obs={obs}, reward={rew}, done={done}, info={info}.")
        cum_reward += rew
        env.render()

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1")
    interact(env)
    input('press enter to quit')
