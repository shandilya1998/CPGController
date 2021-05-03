import torch
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate
import torch
from collections import OrderedDict

USE_CUDA = False
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def complex_relu(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            torch.nn.functional.relu(x),
            torch.nn.functional.relu(y)
        ], - 1
    )
    return out

def complex_elu(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            torch.nn.functional.elu(x),
            torch.nn.functional.elu(y)
        ], - 1
    )
    return out

def complex_tanh(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    denominator = torch.cos(2*x) + torch.cosh(2*y)
    x = torch.sin(2*x) / denominator
    y = torch.sinh(2*y) / denominator
    out = torch.cat([x, y], -1)
    return out

def apply_complex(fr, fi, input, dtype = torch.float32):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            fr(x) - fi(y),
            fr(y) + fi(x)
        ], -1
    )
    return out

class ComplexReLU(torch.nn.Module):

     def forward(self,input):
         return complex_relu(input)

class ComplexELU(torch.nn.Module):

     def forward(self,input):
         return complex_elu(input)

class ComplexTanh(torch.nn.Module):

    def forward(self, input):
        return complex_tanh(input)

class ComplexLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = torch.nn.Linear(in_features, out_features)
        self.fc_i = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class ParamNet(torch.nn.Module):
    def __init__(self,
        params,
    ):
        super(ParamNet, self).__init__()
        self.params = params
        motion_seq = OrderedDict()
        input_size = params['motion_state_size']
        output_size_motion_state_enc = None
        for i, units in enumerate(params['units_motion_state']):
            motion_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            motion_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
            input_size = units
            output_size_motion_state_enc = units
        self.motion_state_enc = torch.nn.Sequential(
            motion_seq
        )

        omega_seq = OrderedDict()
        for i, units in enumerate(params['units_omega']):
            omega_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            omega_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
            input_size = units

        omega_seq['out_fc'] = torch.nn.Linear(
            input_size,
            1
        )
        omega_seq['out_ac'] = torch.nn.ReLU()

        self.omega_dense_seq = torch.nn.Sequential(
            omega_seq
        )

        mu_seq = OrderedDict()
        for i, units in enumerate(params['units_mu']):
            mu_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            mu_seq['ac{i}'.format( i = i)] = torch.nn.ELU()
            input_size = units

        mu_seq['out_fc'] = torch.nn.Linear(
            input_size,
            params['units_osc']
        )
        mu_seq['out_ac'] = torch.nn.ReLU()

        self.mu_dense_seq = torch.nn.Sequential(
            mu_seq
        )

    def forward(self, desired_motion):
        x = self.motion_state_enc(desired_motion)
        omega = self.omega_dense_seq(x)
        mu = self.mu_dense_seq(x)
        return omega, mu

class Hopf(torch.nn.Module):
    def __init__(self, params):
        super(Hopf, self).__init__()
        self.params = params
        self.dt = self.params['dt']
        self.arange = torch.arange(0, self.params['units_osc'], 1.0)

    def forward(self, z, omega, mu):
        units_osc = z.shape[-1]
        x, y = torch.split(z, units_osc // 2, -1)
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y,x)
        delta_phi = self.dt * omega * self.arange
        phi = phi + delta_phi
        r = r + self.dt * (mu - r ** 2) * r
        z = torch.cat([x, y], -1)
        return z

class RhythmGenerator(torch.nn.Module):
    def __init__(self, params):
        super(RhythmGenerator, self).__init__()
        self.params = params

        self.hopf = Hopf(self.params)
        complex_seq = OrderedDict()
        input_size = params['units_osc']
        for i, units in enumerate(params['units_output_mlp'][:-1]):
            complex_seq['fc{i}'.format(i = i)] = ComplexLinear(
                input_size,
                units
            )
            complex_seq['ac{i}'.format(i = i)] =  ComplexELU()
            input_size = units

        complex_seq['out_fc'] = ComplexLinear(
            input_size,
            self.params['action_dim']
        )
        complex_seq['out_ac'] = ComplexTanh()
        self.complex_mlp = torch.nn.Sequential(
            complex_seq
        )

    def forward(self, z, mod_state, omega, mu):
        z = self.hopf(z, omega, mu)
        z = z + mod_state
        out = self.complex_mlp(z)
        size = out.shape[-1]//2
        x, y = torch.split(out, size, -1)
        return 2 * x, z, omega, mu

class PretrainCell(torch.nn.Module):
    def __init__(self, params):
        super(PretrainCell, self).__init__()
        self.params = params
        self.param_net = ParamNet(self.params)
        self.rhythm_gen = RhythmGenerator(self.params)

    def forward(self, desired_motion, mod_state, z):
        omega, mu = self.param_net(desired_motion)
        actions, Z, _, _ = self.rhythm_gen(
            z, mod_state, omega, mu
        )
        return actions, omega, mu, Z

class ActorCell(torch.nn.Module):
    def __init__(self, params, cell = None):
        super(ActorCell, self).__init__()
        self.params = params
        if cell is None:
            self.pretrain_cell = PretrainCell(self.params)
        else:
            self.pretrain_cell = cell
        self.gru = torch.nn.GRUCell(
            self.params['robot_state_size'],
            self.params['units_robot_state'][0]
        )
        robot_state_enc_seq = OrderedDict()
        input_size = self.params['units_robot_state'][0]
        for i, units in enumerate(self.params['units_robot_state'][1:]):
            robot_state_enc_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            robot_state_enc_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
            input_size = units
        self.robot_state_enc = torch.nn.Sequential(robot_state_enc_seq)
        self.robot_enc_state = torch.autograd.Variable(
            torch.zeros(1, self.params['units_robot_state'][0])
        ).type(FLOAT)
        self.z = torch.autograd.Variable(
            torch.zeros(1, 2 * self.params['units_osc'])
        ).type(FLOAT)

    def reset_gru_hidden_state(self, done=True):
        if done:
            self.robot_enc_state = torch.autograd.Variable(
                torch.zeros(1, self.params['units_robot_state'][0])
            ).type(FLOAT)
            self.z = torch.autograd.Variable(
                torch.zeros(1, 2 * self.params['units_osc'])
            ).type(FLOAT)
        else:
            self.robot_enc_state = torch.autograd.Variable(
                self.robot_enc_state.data
            ).type(FLOAT)
            self.z = torch.autograd.Variable(
                self.z
            ).type(FLOAT)

    def forward(self, desired_motion, robot_state, hidden_state = None):
        robot_enc_state = self.robot_enc_state
        z = self.z
        if hidden_state is not None:
            robot_enc_state, z = hidden_state
        robot_enc_state = self.gru(robot_state, robot_enc_state)
        rs_r = self.robot_state_enc(self.robot_enc_state)
        rs_i = torch.zeros_like(rs_r)
        rs = torch.cat([rs_r, rs_i], -1)
        actions, omega, mu, z = \
            self.pretrain_cell(desired_motion, rs, z)
        if hidden_state is None:
            self.robot_enc_state = robot_enc_state
            self.z = z
        return actions, robot_enc_state, z

class Actor(torch.nn.Module):
    def __init__(self, params, cell = None):
        super(Actor, self).__init__()
        self.params = params
        if cell is None:
            self.cell = ActorCell(self.params)
        else:
            self.cell = cell

    def reset_gru_hidden_state(self, done=True):
        self.cell.reset_gru_hidden_state(done)

    def forward(self, desired_motion, robot_state, hidden_state = None):
        action, robot_enc_state, z = self.cell(
                desired_motion,
                robot_state,
                hidden_state
        )
        return action, (robot_enc_state, z)

class Critic(torch.nn.Module):
    def __init__(self, params):
        super(Critic, self).__init__()
        self.params = params
        motion_state_seq = OrderedDict()
        input_size = self.params['motion_state_size']
        output_size_motion_state = None
        for i, units in enumerate(self.params['units_motion_state_critic']):
            motion_state_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            input_size = units
            output_size_motion_state = units
            motion_state_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
        self.motion_state_seq = torch.nn.Sequential(
            motion_state_seq
        )

        robot_state_seq = OrderedDict()
        input_size = self.params['robot_state_size']
        output_size_robot_state = None
        for i, units in enumerate(self.params['units_robot_state_critic']):
            robot_state_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            input_size = units
            output_size_robot_state = units
            robot_state_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
        self.robot_state_seq = torch.nn.Sequential(
            robot_state_seq
        )

        action_state_seq = OrderedDict()
        input_size = self.params['action_dim']
        output_size_action_state = None
        for i, units in enumerate(self.params['units_action_critic']):
            action_state_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            input_size = units
            output_size_action_state = units
            action_state_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
        self.action_seq = torch.nn.Sequential(
            action_state_seq
        )

        out_dense_seq = []
        out_dense_seq.append(torch.nn.Linear(
            output_size_action_state + output_size_robot_state + \
                output_size_motion_state,
            1,
        ))
        out_dense_seq.append(torch.nn.ELU())
        self.out_dense_seq = torch.nn.Sequential(
            *out_dense_seq
        )

    def forward(self, desired_motion, robot_state, actions):
        ms = self.motion_state_seq(desired_motion)
        rs = self.robot_state_seq(robot_state)
        ac = self.action_seq(actions)
        q = self.out_dense_seq(torch.cat([ms, rs, ac], -1))
        return q

