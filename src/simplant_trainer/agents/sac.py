from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.hyperparameters import HyperParameters
from simplant_trainer.value_objects.metrics import EpochMetrics
from simplant_trainer.value_objects.transition import Transition 
from simplant_trainer.agents.buffer import BoundedReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from safetensors.torch import save, load
from numpy import ndarray

# Actor Neural Network
class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)

        u = dist.rsample()
        action = torch.tanh(u)

        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, obs):
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.qnet = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.qnet(x).squeeze(-1)
    
    
class SACAgent(nn.Module):
    def __init__(self, env_metadata: EnvMetadata, hp: HyperParameters):
        super().__init__()
        self.hp = hp

        # Limits and Dimensions
        self.obs_dim = env_metadata.observation_space.shape[0]
        self.action_dim = env_metadata.action_space.shape[0]
        self.action_low = torch.tensor(env_metadata.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(env_metadata.action_space.high, dtype=torch.float32)

        # Networks
        self.actor = SquashedGaussianActor(self.obs_dim, self.action_dim)
        self.q1 = QNetwork(self.obs_dim, self.action_dim)
        self.q2 = QNetwork(self.obs_dim, self.action_dim)       

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Targets never receives gradients. The updated just by Polyak averaging
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        # Optimizers and Alpha
        lr = hp.learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -self.action_dim # That's the SAC heuristic: target entropy is minus the dimensionality of actions.

        self.gamma = hp.gamma
        self.tau = hp.extras.get("tau", 0.005)
        self.batch_size = hp.batch_size

        self.buffer = BoundedReplayBuffer(hp.buffer_capacity, (self.obs_dim,), (self.action_dim,))
         
    def select_action(self, observation: ndarray, deterministic: bool = False) -> ndarray:
        state = torch.FloatTensor(observation).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        
        action = action.squeeze(0)
        real_action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        return real_action.cpu().numpy()

    def observe(self, transition: Transition):
        self.buffer.add(transition)

    def update(self) -> EpochMetrics | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch["next_state"])
            q1_t = self.q1_target(batch["next_state"], next_action)
            q2_t = self.q2_target(batch["next_state"], next_action)
            q_target = torch.min(q1_t, q2_t)

            alpha = self.log_alpha.exp()

            """
                Descomponiendo esta línea:

                - batch["rewards"]: la recompensa inmediata que obtuviste
                - self.gamma * (...): el valor futuro descontado
                - (1 - batch["dones"]): si el episodio terminó, el futuro vale 0
                - q_target_min: el valor estimado del siguiente estado
                - alpha * next_log_prob: el bonus de entropía. next_log_prob es negativo (es un log de probabilidad), así que restar alpha por un negativo suma valor cuando la acción es más aleatoria
            """
            y = batch["rewards"] + self.gamma * (1 - batch["dones"]) * (q_target - alpha * next_log_prob)

            # Update critics
        q1_loss = F.mse_loss(self.q1(batch["obs"], batch["actions"]), y)
        q2_loss = F.mse_loss(self.q2(batch["obs"], batch["actions"]), y)
        for opt, loss in [(self.q1_optimizer, q1_loss), (self.q2_optimizer, q2_loss)]:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        
        for p in self.q1.parameters():
            p.requires_grad_(False)
        for p in self.q2.parameters():
            p.requires_grad_(False)

        new_action, log_prob = self.actor.sample(batch["obs"])
        q1_new = self.q1(batch["obs"], new_action)
        q2_new = self.q2(batch["obs"], new_action)
        q_min = torch.min(q1_new, q2_new)

        actor_loss = (alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.q1.parameters():
            p.requires_grad_(True)
        for p in self.q2.parameters():
            p.requires_grad_(True)       

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()       

        # Polyak update de targets
        # Sin gradientes, pura aritmética sobre los pesos:
        with torch.no_grad():
            for p_target, p_online in zip(self.q1_target.parameters(), self.q1.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p_online.data)

            for p_target, p_online in zip(self.q2_target.parameters(), self.q2.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p_online.data)

        return EpochMetrics(
            epoch=0,  # el Session lo sobreescribe
            mean_reward=0.0,  # el Session lo calcula
            policy_loss=actor_loss.item(),
            value_loss=(q1_loss.item() + q2_loss.item()) / 2,
            entropy=max(0.0, -log_prob.mean().item()),
        )

    def save_state(self) -> bytes:
        """
        Serializa el estado completo del agente. 
        Fundamento: Transformar estructuras jerárquicas (redes) en un mapa plano de tensores.
        """
        tensors = {}

        # 7.1 — Empaquetado con prefijos
        # Diccionario de mapeo para iterar las redes
        networks = {
            "actor": self.actor,
            "q1": self.q1,
            "q2": self.q2,
            "q1_target": self.q1_target,
            "q2_target": self.q2_target
        }

        for prefix, net in networks.items():
            for key, val in net.state_dict().items():
                tensors[f"{prefix}.{key}"] = val

        # 7.2 — Guardar parámetros adicionales
        tensors["log_alpha"] = self.log_alpha.detach()

        # 7.3 — Conversión a bytes (formato safetensors)
        return save(tensors)

    def load_state(self, state: bytes):
        """
        Restaura el estado y recrea los optimizadores.
        """
        # 8.1 — Carga de tensores planos
        flat_tensors = load(state)

        # 8.2 — Reconstrucción de diccionarios de estado
        prefixes = ["actor", "q1", "q2", "q1_target", "q2_target"]
        state_dicts = {p: {} for p in prefixes}

        for key, tensor in flat_tensors.items():
            if key == "log_alpha":
                # Restaurar log_alpha asegurando que sea un parámetro con gradiente
                self.log_alpha = tensor.clone().detach().requires_grad_(True)
                continue

            # Separar el prefijo del nombre del parámetro (ej: 'actor.layer.weight' -> 'actor', 'layer.weight')
            prefix, real_key = key.split(".", 1)
            if prefix in state_dicts:
                state_dicts[prefix][real_key] = tensor

        # 8.3 — Cargar pesos en las redes
        self.actor.load_state_dict(state_dicts["actor"])
        self.q1.load_state_dict(state_dicts["q1"])
        self.q2.load_state_dict(state_dicts["q2"])
        self.q1_target.load_state_dict(state_dicts["q1_target"])
        self.q2_target.load_state_dict(state_dicts["q2_target"])

        # 8.4 — Recrear optimizadores
        # Se pierden los momentos (Adam), pero se recupera la funcionalidad sin dependencias de pickle
        lr = self.hp.learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def close(self):
        """
        Elimina referencias y limpia la caché de la GPU.
        """
        # 8.1 — Nulificar redes para facilitar la recolección de basura (GC)
        self.actor = None
        self.q1 = None
        self.q2 = None
        self.q1_target = None
        self.q2_target = None

        # 8.2 — Vaciar el buffer (que suele ser el objeto más pesado en RAM)
        if hasattr(self, 'buffer'):
            self.buffer = None

        # 8.3 — Limpieza de VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
