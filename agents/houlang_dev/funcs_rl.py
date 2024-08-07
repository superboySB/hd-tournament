import os
import json
import numpy as np
import collections
import cloudpickle
import torch
from torch import nn

def get_activation(act_name, **kwargs):
  activations = {
    None: lambda x: x,
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
  }
  if isinstance(act_name, str):
    act_name = act_name.lower()
  assert act_name in activations, act_name
  return activations[act_name]


def tree_map(func, tree, *args, is_leaf=None, level=None):
  if tree is None:
    return tree
  if is_leaf is not None and is_leaf(tree):
    return func(tree, *args)
  if level == 0:
    return func(tree, *args)
  if level is not None:
    level -= 1
  if isinstance(tree, (list, tuple)):
    if hasattr(tree, '_fields'):
      return type(tree)(*[tree_map(func, *x, is_leaf=is_leaf, level=level) 
                          for x in zip(tree, *args)])
    else:
      return type(tree)(tree_map(func, *x, is_leaf=is_leaf, level=level) 
                        for x in zip(tree, *args))
  elif isinstance(tree, dict):
    return type(tree)({k: tree_map(func, v, *[a[k] for a in args], is_leaf=is_leaf, level=level) 
                       for k, v in tree.items()})
  else:
    return func(tree, *args)


def _prepare_for_rnn(x):
  if x is None:
    return x, None
  x = x.transpose(0, 1)
  shape = x.shape
  x = x.reshape(x.shape[0], -1, *x.shape[3:])
  return x, shape


def _recover_shape(x, shape):
  x = x.reshape(*shape[:3], x.shape[-1])
  x = x.transpose(0, 1)
  return x


def _prepare_rnn_state(state, shape):
  state = tree_map(
    lambda x: x.permute(2, 0, 1, 3).reshape(shape), state
  )
  return state


def _recover_rnn_state(state, shape):
  state = tree_map(
    lambda x: x.permute(1, 0, 2).reshape(shape), state
  )
  return state


class MLP(nn.Module):
  def __init__(
    self, 
    input_dim, 
    units_list=[], 
    out_size=None, 
    activation=None, 
    w_init='glorot_uniform', 
    b_init='zeros', 
    name=None, 
    out_scale=1, 
    norm=None, 
    norm_after_activation=False, 
    norm_kwargs={
      'elementwise_affine': True, 
    }, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    rnn_type=None, 
    rnn_layers=1, 
    rnn_units=None, 
    rnn_init='orthogonal',
    rnn_norm=False, 
  ):
    super().__init__()
    units_list = [input_dim] + units_list
    self.layers = nn.Sequential()
    for i, u in enumerate(units_list[1:]):
      layers = nn.Sequential()
      l = nn.Linear(units_list[i], u)
      layers.append(l)
      if norm == 'layer' and not norm_after_activation:
        layers.append(nn.LayerNorm(u, **norm_kwargs))
      layers.append(get_activation(activation))
      if norm == 'layer' and norm_after_activation:
        layers.append(nn.LayerNorm(u, **norm_kwargs))
      self.layers.append(layers)

    self.rnn_type = rnn_type
    self.rnn_layers = rnn_layers
    self.rnn_units = rnn_units
    self.rnn = None
    input_dim = u
    
    if out_size is not None:
      self.out_layer = nn.Linear(input_dim, out_size)
    else:
      self.out_layer = None

  def forward(self, x, reset=None, state=None):
    x = self.layers(x)
    if self.out_layer is not None:
        x = self.out_layer(x)
    return x


class CategoricalOutput(nn.Module):
  def __init__(
    self, 
    num_inputs, 
    num_outputs, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=0.01
  ):
    super().__init__()
    self.linear = nn.Linear(num_inputs, num_outputs)
    self.out_w_init = out_w_init
    self.out_b_init = out_b_init
    self.out_scale = out_scale

  def forward(self, x, action_mask=None):
    x = self.linear(x)
    if action_mask is not None:
      x[action_mask == 0] = -1e10
    return x


class MultivariateNormalOutput(nn.Module):
  def __init__(
    self, 
    num_inputs, 
    num_outputs, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=0.01, 
    out_act='tanh', 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    init_std=.2,
    tpdv={'device': 'cpu'},
  ):
    super().__init__()
    self.linear = nn.Linear(num_inputs, num_outputs)
    self.out_act = get_activation(out_act)
    self.sigmoid_scale = sigmoid_scale
    self.std_x_coef = std_x_coef
    self.std_y_coef = std_y_coef
    self.init_logstd = np.log(init_std)
    if sigmoid_scale:
      self.logstd = nn.Parameter(self.std_x_coef + torch.zeros(num_outputs))
    else:
      self.logstd = nn.Parameter(self.init_logstd + torch.zeros(num_outputs))
    self.tpdv = tpdv

  def forward(self, x):
    mean = self.linear(x)
    mean = self.out_act(mean)
    if self.sigmoid_scale:
      scale = torch.sigmoid(self.logstd / self.std_x_coef) * self.std_y_coef
    else:
      scale = torch.exp(self.logstd)
    return mean, scale


def tpdv(device):
  return dict(dtype=torch.float32, device=torch.device(device))


class Policy(nn.Module):
  def __init__(
    self, 
    input_dim, 
    is_action_discrete, 
    action_dim, 
    out_act=None, 
    init_std=.2, 
    sigmoid_scale=True, 
    std_x_coef=1., 
    std_y_coef=.5, 
    use_action_mask={'action': False}, 
    out_w_init='orthogonal', 
    out_b_init='zeros', 
    out_scale=.01, 
    device='cpu', 
    **config
  ):
    super().__init__()
    self.tpdv = tpdv(device)
    self.action_dim = action_dim
    self.is_action_discrete = is_action_discrete

    self.out_act = out_act
    self.use_action_mask = use_action_mask
    self.net = MLP(input_dim, **config)
    self.heads = {}
    for k in action_dim:
      if is_action_discrete[k]:
        self.heads[k] = CategoricalOutput(
          config['rnn_units'], action_dim[k], 
          out_w_init, out_b_init, out_scale)
      else:
        self.heads[k] = MultivariateNormalOutput(
          config['rnn_units'], action_dim[k], 
          out_w_init, out_b_init, out_scale, 
          out_act=out_act, sigmoid_scale=sigmoid_scale, 
          std_x_coef=std_x_coef, std_y_coef=std_y_coef, 
          init_std=init_std, tpdv=self.tpdv)
    for k, v in self.heads.items():
      setattr(self, f'head_{k}', v)

  def forward(self, x, reset=None, state=None, action_mask=None):
    x, reset, state, action_mask = tree_map(
      lambda x: x.to(**self.tpdv), (x, reset, state, action_mask))
    x = self.net(x, reset, state)
    if isinstance(x, tuple):
      assert len(x) == 2, x
      x, state = x
    
    outs = {}
    for name, layer in self.heads.items():
      if self.use_action_mask.get(name, False):
        assert self.is_action_discrete[name], self.is_action_discrete[name]
        am = action_mask[name]
        d = layer(x, action_mask=am)
      else:
        d = layer(x)
      outs[name] = d
    return outs, state


def expand_dims_match(x: np.ndarray, target: np.ndarray):
  """ Expands dimensions of x to match target,
  an efficient implementation of the following process 
    while len(x.shape) < len(target.shape):
      x = np.expand_dims(x, -1)
  """
  if x.ndim == target.ndim:
    return x
  elif x.shape == target.shape[-x.ndim:]:
    # adding axes to the front
    return x[(*(None,)*(target.ndim - x.ndim), *[slice(None) for _ in x.shape])]
  elif x.shape == target.shape[:x.ndim]:
    # adding axes to the end
    return x[(*[slice(None) for _ in x.shape], *(None,)*(target.ndim - x.ndim))]
  else:
    raise ValueError(f'Incompatible shapes: {(x.shape, target.shape)}')


def normalize(x, mean, std, zero_center=True, clip=None, mask=None, 
        dim_mask=None, np=np):
  """ Normalize x using mean and std
  mask chooses which samples to apply normalization
  dim_mask masks out dimensions with small variance
  """
  x_new = x
  dtype = x.dtype
  if zero_center:
    x_new = x_new - mean
  std = std if dim_mask is None else np.where(dim_mask, std, 1.)
  x_new = x_new / std
  if clip:
    x_new = np.clip(x_new, -clip, clip)
  if mask is not None:
    mask = expand_dims_match(mask, x_new)
    x_new = np.where(mask, x_new, x)
  x = x.astype(dtype)
  return x_new


def load_params(path_dir):
  with open(path_dir, 'rb') as f:
    data = cloudpickle.load(f)
  model_params = data['model']
  rms_params = data['rms']
  return model_params, rms_params


def discrete2continuous(action, n_bins=41):
  assert np.all(action < n_bins), (action, n_bins)
  new_action = action * 2 / (n_bins - 1) - 1
  assert np.all(new_action <= 1) and np.all(new_action >= -1), (action, new_action, n_bins)
  return new_action
  

def get_control_action(action):
  disc_control = np.stack([
    action[Action.AILERON], 
    action[Action.ELEVATOR], 
    action[Action.RUDDER], 
    action[Action.THROTTLE]
  ], -1)
  control = discrete2continuous(disc_control)
  return control


def action2cmd(action):
  if action is None:
    return {}

  act = list(action)
  act[-1] = (act[-1] + 1) / 2
  cmd = {'control': act}
  assert 0 <= cmd['control'][-1] <= 1, (cmd['control'])

  return cmd


def get_obs(info, target_status):
  obs = []
  is_uav = info.is_uav
  delta_altitude = (target_status[0] - info.height) / 1000
  delta_velocity = (target_status[1] - info.sp) / 340
  target_heading = target_status[2]
  current_heading = info.yaw
  if target_heading > np.pi and current_heading < 0:
    current_heading += 2*np.pi
  elif target_heading < -np.pi and current_heading > 0:
    current_heading -= 2*np.pi
  delta_heading = (target_heading - current_heading) / np.pi
  height = info.height / 10000
  roll = info.roll / np.pi
  pitch = info.pitch / np.pi
  aoa = info.alpha
  sideslip = info.beta
  omega = [info.omega_p, info.omega_q, info.omega_r]
  v_north = info.v_north / 340
  v_east = info.v_east / 340 
  v_down = info.v_down / 340  #地向
  v = info.sp / 340    #地速 空速是tas

  obs = np.array([
    is_uav,                 # 0. is_uav           (unit: bool)
    delta_altitude,         # 1. delta_h          (unit: m)
    delta_velocity,         # 2. delta_v          (unit: m/s)
    delta_heading,          # 3. delta_heading    (unit: °)
    height,                 # 4. altitude         (unit: m)
    roll, pitch,            # 5, 6. roll, pitch   (unit: rad)
    aoa, sideslip,          # 7, 8. aoa, sideslip (unit: rad)
    *omega,                 # 9, 10, 11. omega    (unit: rad/s)  
    v_north,                # 12. v_body_x        (unit: m/s)
    v_east,                 # 16. v_body_y        (unit: m/s)
    v_down,                 # 14. v_body_z        (unit: m/s)
    v,                      # 15. vc              (unit: m/s)
  ], np.float32)

  return obs


class FCModel:
  def __init__(self, path, obs_dim, is_action_discrete, action_dim):
    model_params, rms_params = load_params(path)
    self.policy = Policy(
      obs_dim, is_action_discrete, action_dim, 
      units_list=[64, 64], activation='relu', rnn_units=64)
    self.policy.load_state_dict(model_params)
    self.is_action_discrete = is_action_discrete
    self.rms_mean = rms_params.mean
    self.rms_std = np.sqrt(rms_params.var + 1e-8)
    # print('fc rms mean', self.rms_mean)
    # print('fc rms std', self.rms_std)

  def __call__(self, obs):
    obs = obs.astype(np.float32)
    # print('fc obs', obs)
    obs = normalize(obs, self.rms_mean, self.rms_std, clip=10)
    # print('fc norm obs', obs)
    obs = torch.from_numpy(obs)
    out, _ = self.policy(obs)
    outs = {}
    for k, v in self.is_action_discrete.items():
      if v:
        outs[k] = torch.distributions.Categorical(logits=out[k]).sample()
      else:
        outs[k] = torch.distributions.Normal(*out[k]).sample()
    return outs
  
  def clip_target(self, target, target_limit=[[1000, 300], [14500, 450]]):
    target[:2] = np.clip(target[:2], target_limit[0], target_limit[1])
    assert np.all(target[:2] >= target_limit[0]), target
    return target

  def get_current_status(self, info):
    current_status = np.zeros(3)
    current_status[0] = info.height
    current_status[1] = info.sp
    current_status[2] = info.yaw
    return current_status

  def get_target(self, info, delta):
    current_status = self.get_current_status(info)
    target = current_status + delta
    target = self.clip_target(target)
    return target
  
  def control_cmd_from_obs(self, obs):
    outs = self(obs)
    outs = tree_map(lambda x: x.numpy(), outs)
    # print('outs', outs)
    control = get_control_action(outs)
    cmd = action2cmd(control)
    return cmd

  def control_cmd(self, info, target):
    obs = get_obs(info, target)
    cmd = self.control_cmd_from_obs(obs)
    return cmd


class Action:
  AILERON = 'action_aileron'
  ELEVATOR = 'action_elevator'
  RUDDER = 'action_rudder'
  THROTTLE = 'action_throttle'


DISC_ACTIONS = set([getattr(Action, k) for k in dir(Action) if not k.startswith('__')])


def create_fc_model(path):
  is_action_discrete = {k: True for k in DISC_ACTIONS}
  action_dim = {k: 41 for k in DISC_ACTIONS}
  model = FCModel(path, 16, is_action_discrete, action_dim)
  return model