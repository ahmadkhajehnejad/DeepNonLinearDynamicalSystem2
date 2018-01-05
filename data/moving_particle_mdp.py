import numpy as np
from PIL import Image, ImageDraw


class MovingParticleMDP(object):
    def __init__(self, H=100, W=100):
        super(MovingParticleMDP, self).__init__()
        self.H, self.W = H, W

        self.agent_size = 6
        self.action_dim = 2
        self.v_min = np.array([-20,-20])
        self.v_max = np.array([20,20])
        self.arange_min = np.array([-10, -10])
        self.arange_max = np.array([10, 10])
        self.anoiserange_min = np.array([-1,-1])
        self.anoiserange_max = np.array([1,1])
        self.onoiserange_min = np.array([-1,-1])
        self.onoiserange_max = np.array([1,1])

#        self.obstacles = np.array([[2, 1], [1, 2], [2, 3], [2, 1.5], [3, 2], [2, 2.5]])
#        self.obstacles[:, 0] = (self.obstacles[:, 0] - 2.5) * 10 + 25
#        self.obstacles[:, 1] = (self.obstacles[:, 1] - 2.5) * 15 + 27.5
        self.im = Image.new('L', (W, H))
        self.draw = ImageDraw.Draw(self.im)
        self.lambda_decay = 0.7

#    def reward_function(self, s, a):
#        return -1, False

    def transition_function(self, current_state, a):
        new_v = self.lambda_decay * current_state[2:4] + a
        next_state = np.zeros(4)
        anoise = np.array([np.random.uniform(self.anoiserange_min[0], self.anoiserange_max[0]),
                          np.random.uniform(self.anoiserange_min[1], self.anoiserange_max[1])])
        next_state[:2] = current_state[:2] + new_v + anoise
        next_state[2:4] = new_v
        onoise = np.array([np.random.uniform(self.onoiserange_min[0], self.onoiserange_max[0]),
                          np.random.uniform(self.onoiserange_min[1], self.onoiserange_max[1])])
        return [next_state, self.render(next_state[:2] + onoise)]

    def sample_random_state(self, shape='rectangle'):
        s = np.array([np.random.uniform(0,self.H), np.random.uniform(0,self.W),
                         np.random.uniform(self.v_min[0],self.v_max[0]),
                         np.random.uniform(self.v_min[1],self.v_max[1]),])
        onoise = np.array([np.random.uniform(self.onoiserange_min[0], self.onoiserange_max[0]),
                          np.random.uniform(self.onoiserange_min[1], self.onoiserange_max[1])])
        return [s, self.render(s[:2] + onoise)]

    def is_valid_state(self, s):
        return (0 <= s[0] <= self.H) and (0 <= s[1] <= self.W) \
                and (self.v_min[0] <= s[2] <= self.v_max[0]) and \
                (self.v_min[1] <= s[3] <= self.v_max[1])

    def render(self, pos, shape='rectangle'):
        self.draw.rectangle((0, 0, self.W, self.H), fill=0)

        # draw obstacles
        '''
        if shape == 'rectangle':
            for obs in self.obstacles:
                x_start = obs[1] - 2
                x_end = obs[1] + 2
                y_start = obs[0] - 2
                y_end = obs[0] + 2
                self.draw.ellipse((x_start, y_start, x_end, y_end), fill=255)
        '''

        # draw agent
        if shape == 'rectangle':
            self.draw.rectangle((pos[1] - self.agent_size/2, pos[0] - self.agent_size/2, pos[1] + self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
        elif shape == 'cross':
            #self.draw.pieslice((pos[1] - 3, pos[0] - 3, pos[1] + 3, pos[0] + 3), 135, 225, fill=255)
            #self.draw.ellipse((pos[1] - self.agent_size/2, pos[0] - self.agent_size/2, pos[1] + self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] + self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] - self.agent_size/2, pos[0] - self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] + self.agent_size/2, pos[0] - self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] - self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
        else:
            print('Not Recognized Shape ' + shape)
            

        return np.asarray(self.im) / 255.0
