#
# Policy interface and parametric policy classes.
#
# @contactrika
#
import numpy as np


class Policy:
    def __init__(self):
        pass

    def get_action(self, obs, t):
        pass

    def get_params(self):
        pass

    def set_params(self, params):
        pass

    def resample_params(self):
        pass

    def scale_params(self, params):
        pass

    def unscale_params(self, scaled_params):
        pass

    def check_params(self, params):
        pass

    @staticmethod
    def scale_params_ranges(params, lows, highs):  # raw -> [0,1]
        return (params - lows) / (highs-lows)

    @staticmethod
    def unscale_params_ranges(scaled_params, lows, highs):  # [0,1] -> raw
        return scaled_params*(highs-lows) + lows

    @staticmethod
    def check_scaled_params(scaled_params):
        assert((scaled_params>=0).all())
        assert((scaled_params<=1).all())

    @staticmethod
    def check_params_ranges(params, lows, highs):
        assert((params>=lows).all())
        assert((params<=highs).all())
        pass


class StructuredPolicy(Policy):
    """
    dim: dimensionality of the controller
    controller_fn: function that constructs a structured controller from a
                   torch array in [0,1]
    """
    def __init__(self, controller_class, controller_dim, t_max, robot,
                 get_init_pos_fxn=None):
        super(StructuredPolicy).__init__()
        self.controller_dim = controller_dim
        self.t_max = t_max
        self.params = np.zeros(controller_dim)
        self.controller = None
        self.controller_class = controller_class
        self.robot = robot
        self.get_init_pos_fxn = get_init_pos_fxn
        self.lows = None; self.highs = None  # assume params already in [0,1]
        self.resample_params()  # init params and controller

    def get_action(self, obs, t=None):
        action = self.controller.get_action(obs, t)
        return action

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.check_params(params)  # check that scaled params were passed
        self.params[:] = params  # copy given params
        self.controller = self.controller_class(
            params, self.t_max, self.robot, self.get_init_pos_fxn)

    def scale_params(self, params):  # raw -> [0,1]
        return params  # assume already scaled

    def unscale_params(self, scaled_params):  # [0,1] -> raw
        return scaled_params # already in [0,1]

    def check_params(self, params):
        assert((params>=0).all())
        assert((params<=1).all())

    def resample_params(self):
        self.params = np.random.rand(*self.params.shape)
        self.controller = self.controller_class(
            self.params, self.t_max, self.robot, self.get_init_pos_fxn)

    def print(self):
        print('StructuredPolicy:', self.controller_class)
        if hasattr(self.controller, 'print'): self.controller.print()
