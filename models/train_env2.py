"""
train_env2.py — Entrenamiento SAC para FlareSystemEnv SIN FGRS
==================================================================
Environment: FlareSystemEnv (19 obs, 3 acciones delta-setpoint)
Agente: SAC con buffer interno (simplant_trainer)

Uso:
    python models/train_env2.py
"""

import numpy as np
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# === Environment ===
from simplant_env.flare_env2 import FlareSystemEnv

# === simplant_trainer ===
from simplant_trainer.agents.sac import SACAgent
from simplant_trainer.value_objects.hyperparameters import HyperParameters
from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.spaces import ObservationSpace, ActionSpace
from simplant_trainer.value_objects.transition import Transition

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
NUM_EPOCHS = 200  # producción: 200
STEPS_PER_EPOCH = 2000  # producción: 2000
WARMUP_STEPS = 5000  # producción: 5000
EVAL_EVERY = 5
EVAL_EPISODES = 2
DEPLOYMENT_EPISODES = 60
RESULTS_DIR = Path("results/results_flare_env2")
RESULTS_DIR.mkdir(exist_ok=True)

OBS_FEATURE_NAMES = [
    "flow_in_norm",
    "nhv_vg_norm",
    "header_pressure_norm",
    "header_temp_norm",
    "nhv_cz_norm",
    "ce",
    "seal_level",
    "ko_level",
    "purge_norm",
    "steam_norm",
    "pilot",
    "h2s_norm",
    "noise_norm",
    "upstream_p_norm",
    "sgr_norm",
    "episode_progress",
    "err_steam_norm",
    "err_purge_norm",
    "err_kod_norm",
]

HP = HyperParameters(
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=128,
    buffer_capacity=50000,
    extras={
        "tau": 0.005,
        "hidden_dim": 256,
        "num_hidden": 2,
        "init_alpha": 1.0,
        "alpha_lr": 3e-4,
    },
)

# Política conservadora para modo delta-setpoint: mantener SPs iniciales
# (operador no interviene) => deltas nulos en todos los canales.
CONSERVATIVE_ACTION = np.zeros(3, dtype=np.float32)


# =========================================================================
# HELPERS
# =========================================================================
def build_metadata(env):
    return EnvMetadata(
        env_id="FlareSystemEnv-noFGRS",
        observation_space=ObservationSpace(
            shape=env.observation_space.shape,
            low=env.observation_space.low,
            high=env.observation_space.high,
        ),
        action_space=ActionSpace(
            shape=env.action_space.shape,
            low=env.action_space.low,
            high=env.action_space.high,
        ),
    )


def evaluate_policy(env, policy_fn, n_episodes=2, deterministic=True):
    """Evalúa una política sobre n episodios completos."""
    results = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=100 + i)
        ep_r, steps = 0.0, 0
        while True:
            action = policy_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            steps += 1
            if term or trunc:
                break
        m = info["episode_metrics"]
        results.append(
            {
                "reward": ep_r,
                "hours": info["time_hours"],
                "ce": m.get("mean_ce", 0),
                "co2eq": m["total_co2eq_kg"],
                "flared_kg": m["total_flared_kg"],
                "esd": info.get("esd_triggered", False),
            }
        )
    return {
        "mean_reward": np.mean([r["reward"] for r in results]),
        "mean_ce": np.mean([r["ce"] for r in results]),
        "mean_co2eq": np.mean([r["co2eq"] for r in results]),
        "mean_flared": np.mean([r["flared_kg"] for r in results]),
        "survival": sum(1 for r in results if not r["esd"]) / len(results) * 100,
    }


def evaluate_detailed(env, policy_fn, seed=42):
    """Episodio completo 24h con timeline detallada."""
    obs, _ = env.reset(seed=seed)
    data = defaultdict(list)
    while True:
        action = policy_fn(obs)
        next_obs, r, term, trunc, info = env.step(action)
        controller = info.get("controller", {})
        sp = controller.get("setpoints", {})
        pv = controller.get("pvs", {})
        e = controller.get("errors_norm", [0.0, 0.0, 0.0])
        rc = info.get("reward_components", {})

        data["time_h"].append(info["time_hours"])
        data["obs"].append(np.array(obs, dtype=np.float32))
        data["reward"].append(r)
        data["flared"].append(info["flared_kgh"])
        data["ce"].append(info["ce"])
        data["pressure"].append(info["header_pressure_barg"])
        data["upstream_p"].append(info.get("upstream_p", np.nan))
        data["pcv_flow_kgh"].append(info.get("pcv_flow_kgh", np.nan))
        data["pcv_setpoint"].append(info.get("pcv_setpoint", np.nan))
        data["pcv_opening_pct"].append(info.get("pcv_opening_pct", np.nan))
        data["noise"].append(info["noise_db"])
        data["h2s_gas_conc"].append(float(next_obs[11] * env.H2S_MAX_MGNM3))
        data["nhv_cz"].append(info["nhv_cz"])
        data["psv_event"].append(1.0 if info.get("psv_event") else 0.0)

        data["sp_steam"].append(sp.get("steam_kgh", np.nan))
        data["sp_purge"].append(sp.get("purge_kgh", np.nan))
        data["sp_kod"].append(sp.get("kod_level_sp", sp.get("kod_drain_kgh", np.nan)))

        data["pv_steam"].append(pv.get("steam_kgh", np.nan))
        data["pv_purge"].append(pv.get("purge_kgh", np.nan))
        data["pv_kod"].append(pv.get("kod_level", pv.get("kod_drain_kgh", np.nan)))

        data["e_steam"].append(float(e[0]))
        data["e_purge"].append(float(e[1]))
        data["e_kod"].append(float(e[2]))

        data["a_steam"].append((action[0] + 1) / 2)
        data["a_purge"].append((action[1] + 1) / 2)
        data["a_kod_level"].append((action[2] + 1) / 2)

        # Reflejando la topología matemática corregida del Nivel 2 y 3
        data["r_recovery"].append(rc.get("recovery", 0.0))
        data["r_flaring"].append(rc.get("flaring", 0.0))
        data["r_incomplete"].append(rc.get("incomplete", 0.0))
        data["r_steam"].append(rc.get("steam", 0.0))
        data["r_purge"].append(rc.get("purge", 0.0))
        data["r_smoothness"].append(rc.get("smoothness", 0.0))
        data["r_ambiental_acotado"].append(rc.get("ambiental_acotado", 0.0))
        data["r_esd"].append(rc.get("esd", 0.0))

        obs = next_obs
        if term or trunc:
            break

    out = {}
    for k, v in data.items():
        if k == "obs":
            out[k] = (
                np.vstack(v).astype(np.float32)
                if v
                else np.empty((0, env.observation_space.shape[0]), dtype=np.float32)
            )
        else:
            out[k] = np.array(v)
    return out, info


def _pct(arr, p):
    if len(arr) == 0:
        return 0.0
    return float(np.percentile(arr, p))


def _mean(arr):
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def analyze_agent_behavior(data, env):
    """Resumen operativo: cómo trabajó el agente para obtener sus resultados."""
    psv_mask = data["psv_event"] > 0.5
    no_psv_mask = ~psv_mask

    reward_totals = {
        "recovery": float(np.sum(data["r_recovery"])),
        "flaring": float(np.sum(data["r_flaring"])),
        "incomplete": float(np.sum(data["r_incomplete"])),
        "steam": float(np.sum(data["r_steam"])),
        "purge": float(np.sum(data["r_purge"])),
        "smoothness": float(np.sum(data["r_smoothness"])),
        "ambiental_acotado": float(np.sum(data["r_ambiental_acotado"])),
        "esd": float(np.sum(data["r_esd"])),
    }
    abs_total = float(sum(abs(v) for v in reward_totals.values())) + 1e-8
    reward_share_pct = {k: abs(v) * 100.0 / abs_total for k, v in reward_totals.items()}

    return {
        "sp_mean": {
            "steam": _mean(data["sp_steam"]),
            "purge": _mean(data["sp_purge"]),
            "kod": _mean(data["sp_kod"]),
        },
        "pv_mean": {
            "steam": _mean(data["pv_steam"]),
            "purge": _mean(data["pv_purge"]),
            "kod": _mean(data["pv_kod"]),
        },
        "tracking_mae": {
            "steam": _mean(np.abs(data["sp_steam"] - data["pv_steam"])),
            "purge": _mean(np.abs(data["sp_purge"] - data["pv_purge"])),
            "kod": _mean(np.abs(data["sp_kod"] - data["pv_kod"])),
        },
        "tracking_nmae_pct": {
            "steam": _mean(np.abs(data["e_steam"])) * 100.0,
            "purge": _mean(np.abs(data["e_purge"])) * 100.0,
            "kod": _mean(np.abs(data["e_kod"])) * 100.0,
        },
        "usage": {
            "steam_high_pct": _mean(data["pv_steam"] > 0.7 * env.STEAM_MAX) * 100.0,
            "purge_near_min_pct": _mean(data["pv_purge"] < 1.2 * env.PURGE_MIN) * 100.0,
        },
        "pcv": {
            "setpoint_mean": _mean(data.get("pcv_setpoint", np.array([]))),
            "flow_mean": _mean(data.get("pcv_flow_kgh", np.array([]))),
            "opening_mean": _mean(data.get("pcv_opening_pct", np.array([]))),
            "upstream_p_mean": _mean(data.get("upstream_p", np.array([]))),
        },
        "psv_response": {
            "psv_steps": int(np.sum(psv_mask)),
            "steam_mean_psv": _mean(data["pv_steam"][psv_mask])
            if np.any(psv_mask)
            else 0.0,
            "steam_mean_no_psv": _mean(data["pv_steam"][no_psv_mask])
            if np.any(no_psv_mask)
            else 0.0,
        },
        "reward_totals": reward_totals,
        "reward_share_abs_pct": reward_share_pct,
    }


def estimate_feature_influence(
    agent,
    obs_history,
    feature_names,
    obs_low=None,
    obs_high=None,
    eps_frac=0.03,
    max_samples=120,
):
    """Sensibilidad local con epsilon proporcional al rango por feature."""
    if obs_history.size == 0:
        return []

    n = obs_history.shape[0]
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples).astype(int)
        samples = obs_history[idx].copy()
    else:
        samples = obs_history.copy()

    def _actions(batch):
        return np.array(
            [agent.select_action(o, deterministic=True) for o in batch],
            dtype=np.float32,
        )

    if obs_low is None or obs_high is None:
        low = np.full(samples.shape[1], -1.0, dtype=np.float32)
        high = np.full(samples.shape[1], 1.0, dtype=np.float32)
    else:
        low = np.asarray(obs_low, dtype=np.float32)
        high = np.asarray(obs_high, dtype=np.float32)

    feat_range = np.maximum(high - low, 1e-6)
    eps_vec = np.maximum(feat_range * float(eps_frac), 1e-6)

    influence = []
    for j, name in enumerate(feature_names):
        eps_j = float(eps_vec[j])
        plus = samples.copy()
        minus = samples.copy()
        plus[:, j] = np.clip(plus[:, j] + eps_j, low[j], high[j])
        minus[:, j] = np.clip(minus[:, j] - eps_j, low[j], high[j])

        a_plus = _actions(plus)
        a_minus = _actions(minus)

        deriv = np.abs(a_plus - a_minus) / (2 * eps_j)
        score = float(np.mean(np.sum(deriv, axis=1)))
        channel_sensitivity = np.mean(deriv, axis=0)
        influence.append(
            {
                "feature": name,
                "score": score,
                "steam": float(channel_sensitivity[0]),
                "purge": float(channel_sensitivity[1]),
                "kod": float(channel_sensitivity[2]),
            }
        )

    influence.sort(key=lambda x: x["score"], reverse=True)
    return influence


def evaluate_deployment(env, policy_fn, n_episodes=DEPLOYMENT_EPISODES, seed_base=1000):
    """Simulación multi-episodio para aproximar comportamiento en despliegue."""
    rows = []
    counters = defaultdict(float)

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        ep_reward = 0.0

        while True:
            action = policy_fn(obs)
            next_obs, r, term, trunc, info = env.step(action)

            ce = float(info["ce"])
            flared = float(info["flared_kgh"])
            nhv_cz = float(info["nhv_cz"])
            header_p = float(info["header_pressure_barg"])
            noise = float(info["noise_db"])
            controller = info.get("controller", {})
            purge_pv = float(controller.get("pvs", {}).get("purge_kgh", 0.0))
            h2s_gas_conc = float(next_obs[11] * env.H2S_MAX_MGNM3)
            pcv_sp = float(info.get("pcv_setpoint", np.nan))
            psv_open = bool(info.get("psv_open", False))

            counters["all_steps"] += 1
            counters["purge_ok"] += 1 if purge_pv >= env.PURGE_MIN else 0
            counters["pressure_op_ok"] += 1 if header_p <= env.P_OP_MAX else 0
            counters["pressure_design_ok"] += 1 if header_p <= env.P_DESIGN else 0
            counters["noise_normal_ok"] += 1 if noise <= env.NOISE_NORMAL else 0
            counters["noise_emerg_ok"] += 1 if noise <= env.NOISE_EMERG else 0
            counters["h2s_ok"] += 1 if h2s_gas_conc <= env.H2S_MAX_MGNM3 else 0
            counters["pcv_sp_ok"] += (
                1 if (env.PCV_SP_MIN <= pcv_sp <= env.PCV_SP_MAX) else 0
            )
            counters["psv_closed"] += 1 if not psv_open else 0

            if flared > 10:
                counters["flare_steps"] += 1
                counters["ce_ok"] += 1 if ce >= env.CE_TARGET else 0
                counters["nhv_ok"] += 1 if nhv_cz >= env.NHV_CZ_MIN else 0

            ep_reward += r
            obs = next_obs
            if term or trunc:
                break

        m = info["episode_metrics"]
        rows.append(
            {
                "reward": ep_reward,
                "ce": float(m.get("mean_ce", 0.0)),
                "co2eq_t": float(m.get("total_co2eq_kg", 0.0)),
                "flared_kg": float(m.get("total_flared_kg", 0.0)),
                "h2s_total": float(m.get("total_h2s_emitted", 0.0)),
                "esd": 1.0 if info.get("esd_triggered") else 0.0,
            }
        )

    def _metric(name):
        arr = np.array([r[name] for r in rows], dtype=np.float32)
        return {
            "mean": _mean(arr),
            "p50": _pct(arr, 50),
            "p95": _pct(arr, 95),
        }

    denom_steps = max(1.0, counters["all_steps"])
    denom_flare = max(1.0, counters["flare_steps"])

    return {
        "episodes": n_episodes,
        "kpi": {
            "reward": _metric("reward"),
            "ce": _metric("ce"),
            "co2eq_t": _metric("co2eq_t"),
            "flared_kg": _metric("flared_kg"),
            "h2s_total": _metric("h2s_total"),
            "esd_rate_pct": _mean(np.array([r["esd"] for r in rows])) * 100.0,
            "esd_free_pct": (1.0 - _mean(np.array([r["esd"] for r in rows]))) * 100.0,
        },
        "compliance": {
            "ce_ok_pct": counters["ce_ok"] * 100.0 / denom_flare if denom_flare > 1.0 else 100.0,
            "nhv_ok_pct": counters["nhv_ok"] * 100.0 / denom_flare if denom_flare > 1.0 else 100.0,
            "purge_ok_pct": counters["purge_ok"] * 100.0 / denom_steps,
            "pressure_op_ok_pct": counters["pressure_op_ok"] * 100.0 / denom_steps,
            "pressure_design_ok_pct": counters["pressure_design_ok"]
            * 100.0
            / denom_steps,
            "noise_normal_ok_pct": counters["noise_normal_ok"] * 100.0 / denom_steps,
            "noise_emerg_ok_pct": counters["noise_emerg_ok"] * 100.0 / denom_steps,
            "h2s_ok_pct": counters["h2s_ok"] * 100.0 / denom_steps,
            "pcv_sp_ok_pct": counters["pcv_sp_ok"] * 100.0 / denom_steps,
            "psv_closed_pct": counters["psv_closed"] * 100.0 / denom_steps,
        },
    }


def build_compliance_matrix(deployment_stats):
    c = deployment_stats["compliance"]
    k = deployment_stats["kpi"]
    rules = [
        {
            "source": "40 CFR 60.18",
            "criterion": "NHV en zona de combustión >= 200 BTU/scf (cuando hay flare)",
            "actual": c["nhv_ok_pct"],
            "target": 99.0,
        },
        {
            "source": "Objetivo operacional",
            "criterion": "Eficiencia de combustión >= 98% (cuando hay flare)",
            "actual": c["ce_ok_pct"],
            "target": 95.0,
        },
        {
            "source": "Seguridad de proceso",
            "criterion": "Purge gas >= mínimo de diseño en todo momento",
            "actual": c["purge_ok_pct"],
            "target": 100.0,
        },
        {
            "source": "BD-136001",
            "criterion": "Presión de operación dentro de rango normal (<= 0.7 barg)",
            "actual": c["pressure_op_ok_pct"],
            "target": 95.0,
        },
        {
            "source": "BD-136001",
            "criterion": "Presión por debajo de diseño (<= 3.5 barg)",
            "actual": c["pressure_design_ok_pct"],
            "target": 100.0,
        },
        {
            "source": "Decreto 351/79",
            "criterion": "Ruido por debajo de umbral de emergencia (<= 115 dB)",
            "actual": c["noise_emerg_ok_pct"],
            "target": 99.0,
        },
        {
            "source": "Decreto 351/79",
            "criterion": "Ruido en banda normal (<= 85 dB)",
            "actual": c["noise_normal_ok_pct"],
            "target": 90.0,
        },
        {
            "source": "Especificación interna",
            "criterion": "Concentración H₂S <= 4 mg/Nm³",
            "actual": c["h2s_ok_pct"],
            "target": 99.0,
        },
        {
            "source": "Seguridad de proceso",
            "criterion": "Episodios sin ESD",
            "actual": k["esd_free_pct"],
            "target": 99.0,
        },
        {
            "source": "Operación PCV",
            "criterion": "PCV setpoint dentro de rango operativo [80, 92] kg/cm²-g",
            "actual": c["pcv_sp_ok_pct"],
            "target": 100.0,
        },
        {
            "source": "Operación normal",
            "criterion": "PSV cerrada en operación normal (sin alivio de emergencia)",
            "actual": c["psv_closed_pct"],
            "target": 99.0,
        },
    ]

    for r in rules:
        gap = r["actual"] - r["target"]
        if gap >= 0:
            status = "✅ CUMPLE"
        elif gap >= -3.0:
            status = "⚠️ BORDE"
        else:
            status = "❌ NO CUMPLE"
        r["status"] = status
    return rules


# =========================================================================
# ENTRENAMIENTO
# =========================================================================
def train():
    print("=" * 70)
    print("SimPlant — Entrenamiento SAC - CREADO POR José Martín Rodriguez Y Humberto Alejando Rodriguez")
    print("=" * 70)

    env = FlareSystemEnv()
    meta = build_metadata(env)
    agent = SACAgent(meta, HP)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"  Obs: {env.observation_space.shape} | Act: {env.action_space.shape}")
    print(f"  Parámetros: {n_params:,}")
    print(f"  Épocas: {NUM_EPOCHS} | Steps/época: {STEPS_PER_EPOCH}")
    print(f"  Warmup: {WARMUP_STEPS} | Buffer: {HP.buffer_capacity:,}")

    # --- Historia ---
    history = defaultdict(list)

    # --- Warmup: acciones random para llenar buffer interno ---
    print(f"\n  Warmup ({WARMUP_STEPS} steps)...")
    obs, _ = env.reset(seed=0)
    for _ in range(WARMUP_STEPS):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        agent.observe(
            Transition(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                terminated=term,  # Corrección estricta de MDP en Warmup
            )
        )
        obs = next_obs if not (term or trunc) else env.reset()[0]

    # --- Training loop ---
    print(f"  Entrenando...\n")
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        obs, _ = env.reset()
        epoch_reward = 0.0
        policy_losses, value_losses, entropies = [], [], []

        for step in range(STEPS_PER_EPOCH):
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, term, trunc, _ = env.step(action)

            agent.observe(
                Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    terminated=term,  # Corrección estricta de MDP en Training
                )
            )

            metrics = agent.update()

            if metrics is not None:
                policy_losses.append(metrics.policy_loss)
                value_losses.append(metrics.value_loss)
                entropies.append(metrics.entropy)

            epoch_reward += reward
            obs = next_obs if not (term or trunc) else env.reset()[0]

        # Logging
        history["epoch_reward"].append(epoch_reward / STEPS_PER_EPOCH)
        history["policy_loss"].append(np.mean(policy_losses) if policy_losses else 0)
        history["value_loss"].append(np.mean(value_losses) if value_losses else 0)
        history["entropy"].append(np.mean(entropies) if entropies else 0)
        history["alpha"].append(agent.log_alpha.exp().item())

        # Evaluación periódica
        if epoch % EVAL_EVERY == 0:
            ev = evaluate_policy(
                env, lambda o: agent.select_action(o, deterministic=True), EVAL_EPISODES
            )
            history["eval_reward"].append(ev["mean_reward"])
            history["eval_ce"].append(ev["mean_ce"])
            history["eval_co2eq"].append(ev["mean_co2eq"])
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:>3}/{NUM_EPOCHS} │ "
                f"R̄={epoch_reward / STEPS_PER_EPOCH:>+8.1f} │ "
                f"Eval R̄={ev['mean_reward']:>+9.1f} │ "
                f"CE={ev['mean_ce']:.3f} │ "
                f"CO₂eq={ev['mean_co2eq']:.1f}t │ "
                f"α={agent.log_alpha.exp().item():.3f} │ {elapsed:.0f}s"
            )

    training_time = time.time() - t0
    total_steps = WARMUP_STEPS + NUM_EPOCHS * STEPS_PER_EPOCH
    print(f"\n  Completado: {total_steps:,} steps en {training_time:.0f}s")

    # --- Guardar modelo ---
    model_path = RESULTS_DIR / "sac_env2_nofgrs_model.safetensors"
    model_path.write_bytes(agent.save_state())
    print(f"  Modelo: {model_path}")

    # --- Evaluación detallada ---
    print(f"\n  Evaluación detallada 24h...")
    sac_data, sac_info = evaluate_detailed(
        env, lambda o: agent.select_action(o, deterministic=True)
    )
    cons_data, cons_info = evaluate_detailed(env, lambda o: CONSERVATIVE_ACTION)

    # --- Análisis de estrategia y compliance para reporte ---
    behavior = analyze_agent_behavior(sac_data, env)
    influence = estimate_feature_influence(
        agent,
        sac_data["obs"],
        OBS_FEATURE_NAMES,
        obs_low=env.observation_space.low,
        obs_high=env.observation_space.high,
    )
    deployment = evaluate_deployment(
        env,
        lambda o: agent.select_action(o, deterministic=True),
        n_episodes=DEPLOYMENT_EPISODES,
    )
    compliance = build_compliance_matrix(deployment)

    # --- Gráficas + reporte enriquecido ---
    generate_plots(
        history,
        sac_data,
        cons_data,
        training_time,
        total_steps,
        behavior,
        influence,
        deployment,
        compliance,
    )

    return agent, history


# =========================================================================
# GRÁFICAS
# =========================================================================
def generate_plots(
    history,
    sac_data,
    cons_data,
    training_time,
    total_steps,
    behavior,
    influence,
    deployment,
    compliance,
):
    C = {
        "sac": "#2563eb",
        "cons": "#94a3b8",
        "good": "#059669",
        "warn": "#f59e0b",
        "bad": "#dc2626",
    }

    # --- 1. Training curves ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "SimPlant (SIN FGRS) — Curvas de Entrenamiento SAC",
        fontsize=15,
        fontweight="bold",
    )
    epochs = range(1, len(history["epoch_reward"]) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history["epoch_reward"], color=C["sac"], linewidth=1.5)
    ax.set_ylabel("Reward/step")
    ax.set_title("Reward por Época")

    ax = axes[0, 1]
    ev_ep = [i * EVAL_EVERY for i in range(1, len(history["eval_reward"]) + 1)]
    ax.plot(
        ev_ep, history["eval_reward"], "s-", color=C["good"], linewidth=2, markersize=6
    )
    ax.set_ylabel("Eval Reward")
    ax.set_title("Reward Evaluación (determinista)")

    ax = axes[0, 2]
    ax.plot(
        ev_ep, [v * 100 for v in history["eval_ce"]], "s-", color=C["sac"], markersize=6
    )
    ax.axhline(98, color=C["good"], linestyle="--", alpha=0.5, label="Objetivo 98%")
    ax.set_ylabel("CE (%)")
    ax.set_title("Eficiencia de Combustión")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, history["policy_loss"], color=C["warn"], linewidth=1)
    ax.set_ylabel("Policy loss")
    ax.set_title("Pérdida del Actor")

    ax = axes[1, 1]
    ax.plot(epochs, history["value_loss"], color=C["bad"], linewidth=1)
    ax.set_ylabel("Q-value loss")
    ax.set_title("Pérdida del Critic")

    ax = axes[1, 2]
    ax.plot(epochs, history["alpha"], color=C["sac"], linestyle="--", linewidth=1.5)
    ax.set_ylabel("α")
    ax.set_title("Temperatura (α)")

    for row in axes:
        for a in row:
            a.set_xlabel("Época")
            a.grid(True, alpha=0.15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    print(f"  → {RESULTS_DIR / 'training_curves.png'}")

    # --- 2. SAC vs Conservador 24h ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(
        "Agente SAC vs Conservador — 24h SIN FGRS", fontsize=15, fontweight="bold"
    )

    t_s, t_c = sac_data["time_h"], cons_data["time_h"]

    ax = axes[0, 0]
    ax.plot(
        t_c, cons_data["ce"] * 100, color=C["cons"], linewidth=1, label="Conservador"
    )
    ax.plot(
        t_s, sac_data["ce"] * 100, color=C["sac"], linewidth=1.2, label="Agente SAC"
    )
    ax.axhline(98, color=C["good"], linestyle="--", alpha=0.4, label="Objetivo 98%")
    ax.set_ylabel("CE (%)")
    ax.set_title("Eficiencia de Combustión")
    ax.set_ylim([80, 101])
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.fill_between(
        t_c, cons_data["flared"], alpha=0.3, color=C["cons"], label="Conservador"
    )
    ax.fill_between(
        t_s, sac_data["flared"], alpha=0.3, color=C["sac"], label="Agente SAC"
    )
    ax.set_ylabel("Flujo (kg/h)")
    ax.set_title("Gas al Flare")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(t_s, sac_data["a_steam"] * 100, color=C["sac"], linewidth=1, label="Steam")
    ax.plot(t_s, sac_data["a_purge"] * 100, color=C["warn"], linewidth=1, label="Purge")
    ax.plot(
        t_s,
        sac_data["a_kod_level"] * 100,
        color=C["good"],
        linewidth=1,
        alpha=0.8,
        label="KOD level",
    )
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    # Reemplazamos "h2s" que ya no existe como r_h2s por los datos del estado de h2s emitido
    # NOTA: En sac_data guardamos h2s emitido en sac_data["h2s"]
    h2s_c = np.cumsum(cons_data.get("h2s", np.zeros_like(t_c))) * 5 / 60
    h2s_s = np.cumsum(sac_data.get("h2s", np.zeros_like(t_s))) * 5 / 60
    ax.plot(t_c, h2s_c, color=C["cons"], linewidth=1.5, label="Conservador")
    ax.plot(t_s, h2s_s, color=C["sac"], linewidth=1.5, label="Agente SAC")
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel("H₂S acumulado (mg)")
    ax.set_title("Emisiones H₂S (acumuladas)")
    ax.legend(fontsize=9)

    for row in axes:
        for a in row:
            a.grid(True, alpha=0.15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(RESULTS_DIR / "sac_vs_conservative.png", dpi=150, bbox_inches="tight")
    print(f"  → {RESULTS_DIR / 'sac_vs_conservative.png'}")

    # --- 3. Resumen ---
    sac_ce = float(np.mean(sac_data["ce"]))
    cons_ce = float(np.mean(cons_data["ce"]))
    sac_co2 = float(
        np.sum(sac_data["flared"]) * (1 - sac_ce) * 0.896 * 80 * 5 / 60 / 1000
    )
    cons_co2 = float(
        np.sum(cons_data["flared"]) * (1 - cons_ce) * 0.896 * 80 * 5 / 60 / 1000
    )

    cons_h2s = float(h2s_c[-1]) if len(h2s_c) else 0.0
    sac_h2s = float(h2s_s[-1]) if len(h2s_s) else 0.0
    co2_improve_pct = 0.0 if cons_co2 <= 1e-8 else (1 - sac_co2 / cons_co2) * 100
    h2s_improve_pct = 0.0 if cons_h2s <= 1e-8 else (1 - sac_h2s / cons_h2s) * 100

    top_influence = influence[:8]

    compliance_lines = []
    for r in compliance:
        compliance_lines.append(
            f"  {r['status']:11s} | {r['source']:<18s} | "
            f"Objetivo {r['target']:>6.1f}% | Resultado {r['actual']:>6.1f}% | {r['criterion']}"
        )

    def _reward_line(key, label):
        total = float(behavior["reward_totals"].get(key, 0.0))
        share = float(behavior["reward_share_abs_pct"].get(key, 0.0))
        effect = "beneficio" if total >= 0.0 else "costo"
        sign = "+" if total >= 0.0 else "-"
        return f"  - {label:<17s}: {share:>5.1f}% ({effect}, {sign}{abs(total):.1f})"

    reward_contribution_lines = "\n".join(
        [
            _reward_line("recovery", "Recuperación"),
            _reward_line("flaring", "Flare"),
            _reward_line("incomplete", "Combustión"),
            _reward_line("steam", "Steam"),
            _reward_line("purge", "Purge"),
            _reward_line("ambiental_acotado", "Ambiental Acotado"),
            _reward_line("esd", "ESD"),
        ]
    )

    report = f"""
{"=" * 70}
SIMPLANT — INFORME 
{"=" * 70}

CONFIGURACIÓN
  Environment:  FlareSystemEnv que se encuentra en el archivo flare_env2.py de simplant_env
  Obs:          19 variables | Act: 3 acciones delta-setpoint
  Épocas:       {NUM_EPOCHS} | Steps: {total_steps:,}
  Tiempo:       {training_time:.0f}s

RESULTADOS (episodio 24h, seed=42)
                    Conservador     Agente SAC      Δ
  CE promedio:      {cons_ce:.3f}           {sac_ce:.3f}           {(sac_ce - cons_ce) * 100:+.1f} pp
  CO₂eq (ton):     {cons_co2:.1f}          {sac_co2:.1f}          {co2_improve_pct:+.1f}%
  H₂S (mg):        {cons_h2s:.1f}          {sac_h2s:.1f}           {h2s_improve_pct:+.1f}%

CÓMO TRABAJÓ EL MODELO (episodio SAC 24h)
  Setpoints promedio:
    Steam={behavior["sp_mean"]["steam"]:.1f} kg/h | Purge={behavior["sp_mean"]["purge"]:.1f} kg/h |
    KOD_level_SP={behavior["sp_mean"]["kod"]:.3f} frac

  Variables ejecutadas (PV) promedio:
    Steam={behavior["pv_mean"]["steam"]:.1f} kg/h | Purge={behavior["pv_mean"]["purge"]:.1f} kg/h |
    KOD_level={behavior["pv_mean"]["kod"]:.3f} frac

  Señales upstream/PCV:
    Upstream P media={behavior["pcv"]["upstream_p_mean"]:.2f} kg/cm²-g |
    PCV setpoint media={behavior["pcv"]["setpoint_mean"]:.2f} kg/cm²-g |
    PCV flow medio={behavior["pcv"]["flow_mean"]:.1f} kg/h |
    PCV apertura media={behavior["pcv"]["opening_mean"]:.1f}%

  Tracking SP→PV (NMAE normalizado):
    Steam={behavior["tracking_nmae_pct"]["steam"]:.2f}% | Purge={behavior["tracking_nmae_pct"]["purge"]:.2f}% |
    KOD={behavior["tracking_nmae_pct"]["kod"]:.2f}% | 

  Patrón operativo:
    Steam alto (>70% max): {behavior["usage"]["steam_high_pct"]:.1f}% del tiempo
    Purge cerca de mínimo (<1.2x min): {behavior["usage"]["purge_near_min_pct"]:.1f}% del tiempo

Contribución absoluta al reward (share):
{reward_contribution_lines}

Variables que más influyeron en decisiones (sensibilidad local actor):
{chr(10).join([f"  - {i + 1:>2}. {row['feature']:<20s} score={row['score']:.4f} | a[steam]={row['steam']:.4f} a[purge]={row['purge']:.4f} a[kod]={row['kod']:.4f}" for i, row in enumerate(top_influence)])}

PROYECCIÓN DE DESPLIEGUE (determinista, {deployment["episodes"]} episodios)
  Reward  mean={deployment["kpi"]["reward"]["mean"]:.1f} | p50={deployment["kpi"]["reward"]["p50"]:.1f} | p95={deployment["kpi"]["reward"]["p95"]:.1f}
  CE      mean={deployment["kpi"]["ce"]["mean"]:.3f} | p50={deployment["kpi"]["ce"]["p50"]:.3f} | p95={deployment["kpi"]["ce"]["p95"]:.3f}
  CO₂eq t mean={deployment["kpi"]["co2eq_t"]["mean"]:.2f} | p50={deployment["kpi"]["co2eq_t"]["p50"]:.2f} | p95={deployment["kpi"]["co2eq_t"]["p95"]:.2f}
  Flared  mean={deployment["kpi"]["flared_kg"]["mean"]:.1f} kg | p95={deployment["kpi"]["flared_kg"]["p95"]:.1f} kg
  ESD rate={deployment["kpi"]["esd_rate_pct"]:.2f}% | Episodios sin ESD={deployment["kpi"]["esd_free_pct"]:.2f}%

COMPARACIÓN CONTRA CRITERIOS AMBIENTALES Y DE SEGURIDAD
{chr(10).join(compliance_lines)}

CONCLUSIÓN
  El agente optimiza los flujos base SIN cambiar equipos para minimizar la penalidad ambiental.
  La mejora viene de optimizar el ratio steam/gas para maximizar CE.
  En despliegue, el cumplimiento normativo debe leerse sobre porcentajes de tiempo (no solo un episodio).

NOTAS
  - La comparación normativa es contra criterios representados en el modelo (proxy de compliance).
  - h2s_gas_conc (concentración en gas) se estima desde observación normalizada (h2s_norm * H2S_MAX_MGNM3).
  - upstream_p_norm se normaliza como (upstream_p - 70) / (99 - 70).
  - Este reporte no reemplaza validación regulatoria en planta ni auditoría HSE.

ARCHIVOS
  {RESULTS_DIR}/training_curves.png
  {RESULTS_DIR}/sac_vs_conservative.png
  {RESULTS_DIR}/sac_env2_nofgrs_model.safetensors
  {RESULTS_DIR}/report.txt
{"=" * 70}
"""
    (RESULTS_DIR / "report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    train()