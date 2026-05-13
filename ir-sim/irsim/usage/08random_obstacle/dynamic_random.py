import irsim

env = irsim.make('dynamic_random.yaml', save_ani=False, full=False)  

for i in range(1000):

    env.step()
    env.render(0.05)

    if env.done():
        break

env.end()