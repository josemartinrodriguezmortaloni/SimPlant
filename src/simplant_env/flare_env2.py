"""
SimPlant — Flare-Only Environment (SIN FGRS)
================================================


Calibrado con datos operativos reales:
    - TRKO-261: KOD de Antorcha (D=2400mm, L=11500mm)
    - B-262/63: Bombas evacuación KOD (centrífugas, 92 m³/h)
    - L-264:    Antorcha (24", 47m, 3 pilotos, 4 MM Sm³/d)

Contrato del entorno:
    - Acción: delta-setpoints (3D, [-1, 1])
      [steam, purge, KOD_level_SP]
    - Observación: 19D = 16 variables de proceso [0,1] + 3 errores [-1,1]

Arquitectura de control:
    - steam y purge usan setpoint del agente + control interno
    - upstream_p exógena + PCV causal + PSV determinística por sobrepresión
    - Loop 3: cascada KOD
    - Loop 4: canal de setpoint PCV (despresurización automática por presión)

Marco regulatorio:
    - Principio "cero quema / cero venteo" en operación normal
    - Resolución SE No. 143/98 (venteo en producción de hidrocarburos)
    - Ley Nacional No. 20.284 (calidad de aire)
    - Ley Nacional No. 27.520 + Protocolo de Kioto + Acuerdo de París
    - Decreto No. 351/79 (ruido: 85 dBA continuo 8h, 140 dB impulsivo max)
    - IRAM 4062-1/2 (ruido comunitario)
    - Diseño "cero humo" (no-smoke)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ─────────────────────────────────────────────────────────────
# Composición del gas de antorcha (base: gas natural tratado)
# ─────────────────────────────────────────────────────────────
GAS_COMPOSITION = {
    "N2": 0.006383,
    "CO2": 0.000568,
    "C1": 0.896391,
    "C2": 0.061831,
    "C3": 0.018647,
    "iC4": 0.004475,
    "nC4": 0.005243,
    "iC5": 0.002099,
    "nC5": 0.001427,
    "C6": 0.001711,
    "C7": 0.000960,
    "C8": 0.000265,
}

NHV_PURE = {  # BTU/scf
    "N2": 0.0,
    "CO2": 0.0,
    "C1": 1012.0,
    "C2": 1773.0,
    "C3": 2520.0,
    "iC4": 3252.0,
    "nC4": 3262.0,
    "iC5": 4000.0,
    "nC5": 4010.0,
    "C6": 4756.0,
    "C7": 5503.0,
    "C8": 6250.0,
}

MW = {  # g/mol
    "N2": 28.01,
    "CO2": 44.01,
    "C1": 16.04,
    "C2": 30.07,
    "C3": 44.10,
    "iC4": 58.12,
    "nC4": 58.12,
    "iC5": 72.15,
    "nC5": 72.15,
    "C6": 86.18,
    "C7": 96.0,
    "C8": 107.0,
}

NHV_MIX = sum(GAS_COMPOSITION[c] * NHV_PURE[c] for c in GAS_COMPOSITION)
MW_MIX = sum(GAS_COMPOSITION[c] * MW[c] for c in GAS_COMPOSITION)
RHO_STD = MW_MIX / 23.645  # kg/Sm³ (gas ideal a 0°C, 1 atm)


class FlareSystemEnv(gym.Env):
    """
    Entorno Gymnasium para antorcha SIN FGRS.

    Layout del sistema (P&ID simplificado):
        Emergency relief lines ─→ [TRKO-261 Liquid KOD] ─→ [Flashback Seal Drum] ─→ [L-264 Flare Stack]
                                        │ oil drain               ↑ makeup water        ↑ steam
                                        │ water drain              purge gas             spark ignition
                                        ↓                                                3 pilotos
                                   [B-262/63 Pumps]
    """

    metadata = {"render_modes": ["human"]}

    # ── Flujos máximos ──────────────────────────────────────
    FLOW_IN_MAX = 207_300.0  # kg/h — diseño máximo del header
    PSV_PEAK_FLOW = 103_650.0  # kg/h — flujo pico de PSV

    # ── Upstream separador AP (SB-211/411) ──────────────────
    UPSTREAM_P_MIN = 70.0  # kg/cm²-g
    UPSTREAM_P_OP_MAX = 89.0  # kg/cm²-g
    UPSTREAM_P_MEAN = 79.5  # kg/cm²-g
    UPSTREAM_P_PSV_SET = 99.0  # kg/cm²-g
    UPSTREAM_P_MAX = 110.0  # kg/cm²-g

    # Dinámica de presión upstream (perturbaciones exógenas)
    UPSTREAM_MEAN_REVERSION = 0.12
    UPSTREAM_CYCLE_AMP = 3.0  # kg/cm²-g
    UPSTREAM_NOISE_SIGMA = 0.5  # kg/cm²-g
    UPSTREAM_SLUG_PROB = 0.01  # prob. por step
    UPSTREAM_SLUG_MIN = 5.0  # kg/cm²-g
    UPSTREAM_SLUG_MAX = 15.0  # kg/cm²-g

    # ── PCV (calibrada con MC-136002) ───────────────────────
    PCV_SP_MIN = 80.0  # kg/cm²-g
    PCV_SP_MAX = 92.0  # kg/cm²-g
    PCV_SP_DEFAULT = 85.0  # kg/cm²-g
    PCV_CV = 2590.0  # Cv de la válvula
    PCV_DENSITY_FACTOR = 1.0
    BLANKETING_LOSSES_KGH = 120.0  # kg/h — pérdidas por blanketing
    FLOW_NOISE_SIGMA = 120.0  # kg/h — ruido de medición

    # ── TRKO-261: KOD de Antorcha (DATOS REALES) ───────────
    KOD_DIAMETER_M = 2.400  # m — diámetro interno
    KOD_LENGTH_M = 11.500  # m — longitud total (T/T)
    KOD_VOLUME_M3 = np.pi / 4 * KOD_DIAMETER_M**2 * KOD_LENGTH_M  # ≈ 52.03 m³
    KOD_LIQUID_DENSITY = 700.0  # kg/m³ — densidad condensado típico

    # ── B-262/63: Bombas evacuación KOD (DATOS REALES) ─────
    KOD_PUMP_FLOW_M3H = 92.0  # m³/h por bomba
    KOD_PUMP_COUNT = 2  # B-262 y B-263
    KOD_PUMP_DISCHARGE_P = 3.4  # kg/cm²-g
    KOD_PUMP_POWER_HP = 16.4  # HP por bomba
    KOD_PUMP_HEAD_M = 39.0  # m — altura desarrollada

    # ── L-264: Antorcha (DATOS REALES) ─────────────────────
    FLARE_TIP_DIAMETER_IN = 24  # pulgadas
    FLARE_STACK_HEIGHT_M = 47.0  # m
    FLARE_PILOT_COUNT = 3
    FLARE_CAPACITY_SM3D = 4_000_000.0  # Sm³/d
    FLARE_CAPACITY_KGH = FLARE_CAPACITY_SM3D / 24.0 * RHO_STD  # ≈ 129,333 kg/h

    # ── Header de antorcha ─────────────────────────────────
    HEADER_P_OP_MIN = 0.5  # barg (= kg/cm²-g del KOD)
    HEADER_P_OP_MAX = 0.7  # barg
    HEADER_P_DESIGN = 3.5  # barg (= P.diseño TRKO-261)
    BACKPRESSURE_MAX_BARG = 0.2  # barg — máxima contrapresión aceptable

    # ── Temperaturas ───────────────────────────────────────
    TEMP_OP_NORMAL = 25.0  # °C (T.asp bombas, T.op KOD)
    TEMP_DESIGN_MAX = 90.0  # °C (diseño KOD y bombas)
    TEMP_DESIGN_MIN = -24.0  # °C (diseño KOD)

    # ── Ruido — Decreto 351/79, IRAM 4062-1/2 ─────────────
    NOISE_NORMAL_DBA = 85.0  # dBA — límite continuo 8h
    NOISE_EMERGENCY_DBA = 115.0  # dBA — exposición ~28 seg
    NOISE_MAX_IMPULSIVE = 140.0  # dB — máximo impulsivo absoluto

    # ── Emisiones y seguridad ──────────────────────────────
    H2S_MAX_MGNM3 = 4.0  # mg/Nm³ — spec gas entrada
    PURGE_GAS_MIN = 30.0  # kg/h — mínimo para evitar flashback
    PURGE_GAS_MAX = 300.0  # kg/h
    STEAM_MAX = 3_000.0  # kg/h — máximo inyección vapor

    # Aliases para compatibilidad con scripts legacy
    H2S_MAX = H2S_MAX_MGNM3
    PURGE_MIN = PURGE_GAS_MIN
    P_OP_MAX = HEADER_P_OP_MAX
    P_DESIGN = HEADER_P_DESIGN
    NOISE_NORMAL = NOISE_NORMAL_DBA
    NOISE_EMERG = NOISE_EMERGENCY_DBA

    # ── Parámetros económicos ──────────────────────────────
    STEAM_COST_PER_KG = 0.025  # USD/kg
    CO2_COST_PER_TON = 10.0  # USD/ton CO₂eq
    NHV_CZ_MIN = 200.0  # BTU/scf — mínimo para combustión estable
    CE_TARGET = 0.98  # eficiencia de combustión objetivo

    # ── Simulación ─────────────────────────────────────────
    TIMESTEP_MIN = 5  # minutos por step
    EPISODE_HOURS = 24  # duración del episodio
    SUBSTEPS = 10  # substeps por step
    SUBSTEP_SEC = 30.0  # segundos por substep

    PRESSURE_TAU = 0.25  # constante de tiempo presión header

    # ── Despresurización controlada ────────────────────────
    DEPRESS_ENABLE_P = 1.5  # barg — presión header para activar depress
    DEPRESS_MAX_FLOW_KGH = 5_000.0  # kg/h — capacidad máxima de la válvula

    def __init__(self, render_mode=None):
        super().__init__()

        # 16 procesos + 3 errores
        obs_low = np.array([0.0] * 16 + [-1.0] * 3, dtype=np.float32)
        obs_high = np.array([1.0] * 16 + [1.0] * 3, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, (19,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

        # Acciones del agente: delta-setpoints
        # [steam, purge, KOD_level_SP]
        self.sp_min = np.array([0.0, 0.0, 0.05], dtype=np.float32)
        self.sp_max = np.array(
            [self.STEAM_MAX, self.PURGE_GAS_MAX, 0.75],
            dtype=np.float32,
        )
        self.sp_delta_max = np.array([250.0, 50.0, 0.03], dtype=np.float32)
        self.err_ranges = self.sp_max - self.sp_min

        # ── PID Loops 1..2: directos a actuador ───────────
        # Loop 1 = steam, Loop 2 = purge
        self.pid_direct_kp = np.array([1.05, 1.10], dtype=np.float32)
        self.pid_direct_ki = np.array([0.004, 0.008], dtype=np.float32)
        self.pid_direct_kd = np.array([2.5, 1.4], dtype=np.float32)
        self.act_tau_direct_s = np.array([70.0, 55.0], dtype=np.float32)

        # ── Cascada KOD (Loop 3) ──────────────────────────
        # Tuning conservador: KOD grande (52 m³) → dinámica lenta
        self.outer_kod_kp = 800.0
        self.outer_kod_ki = 0.15
        self.outer_kod_kd = 1500.0

        # ── Lazo presión header → depress valve (Loop interno) ──
        self.outer_pressure_kp = 1.2

        # Lazos internos: SP cascada → actuador real
        # [0] = drenaje KOD (B-262/63), [1] = válvula despresurización
        self.ko_drain_max = (
            self.KOD_PUMP_FLOW_M3H * self.KOD_PUMP_COUNT * self.KOD_LIQUID_DENSITY
        )
        self.ko_drain_min = 0.0
        self.depress_min = 0.0
        self.depress_max = self.DEPRESS_MAX_FLOW_KGH

        self.inner_kp = np.array([1.00, 1.20], dtype=np.float32)
        self.inner_ki = np.array([0.003, 0.015], dtype=np.float32)
        self.inner_kd = np.array([2.0, 0.6], dtype=np.float32)
        self.inner_tau_s = np.array([90.0, 45.0], dtype=np.float32)

        self.state = None
        self.steps = 0
        self.max_steps = int(self.EPISODE_HOURS * 60 / self.TIMESTEP_MIN)
        self.prev_action = None
        self.render_mode = render_mode
        self.np_random = None
        self.episode_metrics = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        init_sp = np.array([200.0, 80.0, 0.10], dtype=np.float32)
        slug_decay_steps = int(self.np_random.integers(3, 9))
        cycle_steps = int(
            self.np_random.integers(
                int(4 * 60 / self.TIMESTEP_MIN),
                int(8 * 60 / self.TIMESTEP_MIN) + 1,
            )
        )
        slug_ttl = int(self.np_random.integers(0, 3))
        slug_amp = float(
            self.np_random.uniform(self.UPSTREAM_SLUG_MIN, self.UPSTREAM_SLUG_MAX)
        )

        self.state = {
            "flow_in": 3500.0,
            "flared_kgh": 3500.0,  # Estado explícito de masa enviada a antorcha
            "nhv_vg": NHV_MIX,
            "header_pressure": 0.5,
            "header_temperature": self.TEMP_OP_NORMAL,
            "steam_flow": 200.0,
            "purge_gas_flow": 80.0,
            "seal_drum_level": 0.5,
            "ko_drum_level": 0.08,
            "nhv_cz": NHV_MIX,
            "combustion_efficiency": 0.99,
            "pilot_flame": 1.0,
            "h2s_in_gas": self.np_random.uniform(1.0, self.H2S_MAX_MGNM3),
            "noise_db": 50.0,
            "upstream_p": self.UPSTREAM_P_MEAN,
            "upstream_phase": self.np_random.uniform(0, 2 * np.pi),
            "upstream_cycle_steps": cycle_steps,
            "slug_amp": slug_amp,
            "slug_ttl": slug_ttl,
            "slug_decay_steps": slug_decay_steps,
            "slug_age": 0,
            "pcv_setpoint_pv": self.PCV_SP_DEFAULT,
            "pcv_opening": 0.0,
            "pcv_flow_kgh": 0.0,
            "psv_open": False,
            "psv_open_prev": False,
            "psv_flow_kgh": 0.0,
            "nhv_phase": self.np_random.uniform(0, 2 * np.pi),
            "esd_triggered": False,
            "controller_sp": init_sp.copy(),
            "direct_pv": np.array([200.0, 80.0], dtype=np.float32),
            "direct_i": np.zeros(2, dtype=np.float32),
            "direct_prev_pv": np.array([200.0, 80.0], dtype=np.float32),
            "outer_i": np.zeros(1, dtype=np.float32),
            "outer_prev_pv": np.array([0.08], dtype=np.float32),
            "inner_sp": np.array([0.0, 0.0], dtype=np.float32),
            "inner_pv": np.array([0.0, 0.0], dtype=np.float32),
            "inner_i": np.zeros(2, dtype=np.float32),
            "inner_prev_pv": np.array([0.0, 0.0], dtype=np.float32),
            "ko_drain_pv": 0.0,
            "depress_pv": 0.0,
        }

        self.steps = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.episode_metrics = {
            "total_flared_kg": 0.0,
            "total_recovered_kg": 0.0,  # siempre 0.0 (SIN FGRS)
            "total_h2s_emitted": 0.0,
            "total_co2eq_kg": 0.0,
            "safety_violations": 0,
            "psv_events": 0,
            "flare_overload_steps": 0,
            "mean_ce": [],
        }

        return self._get_obs(), self._get_info()

    def _pid_step(self, sp, pv, i_old, prev_pv, kp, ki, kd, out_min, out_max):
        err = sp - pv
        d_pv = (pv - prev_pv) / self.SUBSTEP_SEC
        d_term = -kd * d_pv

        i_candidate = i_old + err * self.SUBSTEP_SEC
        unsat = kp * err + ki * i_candidate + d_term
        out = np.clip(unsat, out_min, out_max)

        sat_high = unsat > out_max
        sat_low = unsat < out_min
        block_i = (sat_high and err > 0.0) or (sat_low and err < 0.0)
        i_new = i_old if block_i else i_candidate
        return out, i_new

    def _compute_error_norm(self):
        s = self.state
        err_norm = np.array(
            [
                (s["controller_sp"][0] - s["steam_flow"]) / (self.err_ranges[0] + 1e-8),
                (s["controller_sp"][1] - s["purge_gas_flow"])
                / (self.err_ranges[1] + 1e-8),
                (s["controller_sp"][2] - s["ko_drum_level"])
                / (self.err_ranges[2] + 1e-8),
            ],
            dtype=np.float32,
        )
        return np.clip(err_norm, -1.0, 1.0)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        s = self.state

        # ─── 1) Supervisor RL → delta-setpoints ───────────
        s["controller_sp"] = np.clip(
            s["controller_sp"] + a * self.sp_delta_max,
            self.sp_min,
            self.sp_max,
        )

        # ─── 2) Perturbaciones exógenas de proceso ────────
        s["upstream_phase"] += 2 * np.pi / max(1, s["upstream_cycle_steps"])
        cycle_target = self.UPSTREAM_P_MEAN + self.UPSTREAM_CYCLE_AMP * np.sin(
            s["upstream_phase"]
        )
        drift = self.UPSTREAM_MEAN_REVERSION * (cycle_target - s["upstream_p"])
        noise_up = self.np_random.normal(0.0, self.UPSTREAM_NOISE_SIGMA)
        slug_term = 0.0
        if s["slug_ttl"] > 0:
            decay_steps = max(1, s["slug_decay_steps"])
            slug_term = s["slug_amp"] * np.exp(-2.5 * s["slug_age"] / decay_steps)
            s["slug_ttl"] -= 1
            s["slug_age"] += 1
        elif self.np_random.random() < self.UPSTREAM_SLUG_PROB:
            s["slug_amp"] = float(
                self.np_random.uniform(self.UPSTREAM_SLUG_MIN, self.UPSTREAM_SLUG_MAX)
            )
            s["slug_decay_steps"] = int(self.np_random.integers(3, 9))
            s["slug_ttl"] = s["slug_decay_steps"]
            s["slug_age"] = 0

        s["upstream_p"] = np.clip(
            s["upstream_p"] + drift + noise_up + slug_term,
            self.UPSTREAM_P_MIN,
            self.UPSTREAM_P_MAX,
        )

        s["nhv_phase"] += 2 * np.pi / (8 * 60 / self.TIMESTEP_MIN)
        shift = 0.03 * np.sin(s["nhv_phase"]) + self.np_random.normal(0, 0.004)
        c1 = np.clip(0.896391 + shift, 0.85, 0.93)
        heavy_scale = (1.0 - c1) / (1.0 - 0.896391)
        s["nhv_vg"] = (
            c1 * NHV_PURE["C1"]
            + 0.061831 * heavy_scale * NHV_PURE["C2"]
            + 0.018647 * heavy_scale * NHV_PURE["C3"]
            + 0.009718 * heavy_scale * NHV_PURE["nC4"]
        )
        s["h2s_in_gas"] = np.clip(
            s["h2s_in_gas"] + self.np_random.normal(0, 0.15),
            0.5,
            self.H2S_MAX_MGNM3,
        )

        psv_open = s["upstream_p"] > self.UPSTREAM_P_PSV_SET
        psv_flow = (
            self.PSV_PEAK_FLOW
            * np.clip((s["upstream_p"] - self.UPSTREAM_P_PSV_SET) / 10.0, 0.0, 1.0)
            if psv_open
            else 0.0
        )
        if psv_open and not s["psv_open_prev"]:
            self.episode_metrics["psv_events"] += 1
        s["psv_open_prev"] = bool(psv_open)

        if s["pilot_flame"] == 1.0 and self.np_random.random() < 0.001:
            s["pilot_flame"] = 0.0
        elif s["pilot_flame"] == 0.0 and self.np_random.random() < 0.33:
            s["pilot_flame"] = 1.0

        dt_sub_h = (self.TIMESTEP_MIN / 60.0) / self.SUBSTEPS
        p_alpha = 1.0 - (1.0 - self.PRESSURE_TAU) ** (1.0 / self.SUBSTEPS)

        # ─── 3) Loop de substeps ──────────────────────────
        flared_int = 0.0
        steam_int = 0.0
        purge_int = 0.0
        pcv_flow_int = 0.0
        psv_flow_int = 0.0
        ch4_fug_int = 0.0
        h2s_emit_int = 0.0
        ce_acc = 0.0
        min_purge = float("inf")
        flare_overloaded = False

        for _ in range(self.SUBSTEPS):
            direct_cmd = np.zeros(2, dtype=np.float32)
            d_min = np.array([0.0, 0.0], dtype=np.float32)
            d_max = np.array([self.STEAM_MAX, self.PURGE_GAS_MAX], dtype=np.float32)
            for j in range(2):
                cmd, i_new = self._pid_step(
                    sp=float(s["controller_sp"][j]),
                    pv=float(s["direct_pv"][j]),
                    i_old=float(s["direct_i"][j]),
                    prev_pv=float(s["direct_prev_pv"][j]),
                    kp=float(self.pid_direct_kp[j]),
                    ki=float(self.pid_direct_ki[j]),
                    kd=float(self.pid_direct_kd[j]),
                    out_min=float(d_min[j]),
                    out_max=float(d_max[j]),
                )
                direct_cmd[j] = cmd
                s["direct_i"][j] = i_new
                s["direct_prev_pv"][j] = s["direct_pv"][j]

            alpha_direct = np.clip(self.SUBSTEP_SEC / self.act_tau_direct_s, 0.0, 1.0)
            s["direct_pv"] = s["direct_pv"] + alpha_direct * (
                direct_cmd - s["direct_pv"]
            )
            s["direct_pv"] = np.clip(s["direct_pv"], d_min, d_max)

            kod_level_sp = float(s["controller_sp"][2])
            kod_pv = float(s["ko_drum_level"])
            level_error = kod_pv - kod_level_sp
            d_level = (kod_pv - float(s["outer_prev_pv"][0])) / self.SUBSTEP_SEC
            d_term = self.outer_kod_kd * d_level
            i_cand = float(s["outer_i"][0]) + level_error * self.SUBSTEP_SEC
            unsat = (
                self.outer_kod_kp * level_error + self.outer_kod_ki * i_cand + d_term
            )
            kod_drain_sp = float(np.clip(unsat, self.ko_drain_min, self.ko_drain_max))

            block_i = (unsat > self.ko_drain_max and level_error > 0) or (
                unsat < self.ko_drain_min and level_error < 0
            )
            s["outer_i"][0] = float(s["outer_i"][0]) if block_i else i_cand
            s["outer_prev_pv"][0] = kod_pv
            s["inner_sp"][0] = kod_drain_sp

            pcv_sp = self.PCV_SP_DEFAULT
            s["pcv_setpoint_pv"] = pcv_sp

            header_p = float(s["header_pressure"])
            if header_p > self.DEPRESS_ENABLE_P:
                depress_demand = self.outer_pressure_kp * (
                    header_p - self.DEPRESS_ENABLE_P
                )
                depress_sp = np.clip(depress_demand, 0.0, self.depress_max)
            else:
                depress_sp = 0.0
            s["inner_sp"][1] = depress_sp

            inner_cmd = np.zeros(2, dtype=np.float32)
            i_min = np.array([self.ko_drain_min, self.depress_min], dtype=np.float32)
            i_max = np.array([self.ko_drain_max, self.depress_max], dtype=np.float32)
            for j in range(2):
                cmd, i_new = self._pid_step(
                    sp=float(s["inner_sp"][j]),
                    pv=float(s["inner_pv"][j]),
                    i_old=float(s["inner_i"][j]),
                    prev_pv=float(s["inner_prev_pv"][j]),
                    kp=float(self.inner_kp[j]),
                    ki=float(self.inner_ki[j]),
                    kd=float(self.inner_kd[j]),
                    out_min=float(i_min[j]),
                    out_max=float(i_max[j]),
                )
                inner_cmd[j] = cmd
                s["inner_i"][j] = i_new
                s["inner_prev_pv"][j] = s["inner_pv"][j]

            alpha_inner = np.clip(self.SUBSTEP_SEC / self.inner_tau_s, 0.0, 1.0)
            s["inner_pv"] = s["inner_pv"] + alpha_inner * (inner_cmd - s["inner_pv"])
            s["inner_pv"] = np.clip(s["inner_pv"], i_min, i_max)

            steam = float(s["direct_pv"][0])
            purge = float(s["direct_pv"][1])
            ko_drain = float(s["inner_pv"][0])
            depress_val = float(s["inner_pv"][1])

            delta_p = max(0.0, s["upstream_p"] - pcv_sp)
            pcv_opening = np.clip(
                delta_p / (self.UPSTREAM_P_PSV_SET - self.PCV_SP_MIN), 0.0, 1.0
            )
            pcv_flow = self.PCV_CV * np.sqrt(delta_p + 1e-8) * self.PCV_DENSITY_FACTOR

            flow_in = np.clip(
                pcv_flow
                + psv_flow
                + purge
                + self.BLANKETING_LOSSES_KGH
                + self.np_random.normal(0.0, self.FLOW_NOISE_SIGMA),
                0.0,
                self.FLOW_IN_MAX,
            )

            # SIN FGRS: todo el gas del proceso va a flare
            flared_from_process = flow_in
            flared = flared_from_process + depress_val

            tip_overload_ratio = flared / (self.FLARE_CAPACITY_KGH + 1e-8)
            if tip_overload_ratio > 1.0:
                flare_overloaded = True

            p_eq = self.HEADER_P_OP_MIN + (flow_in / self.FLOW_IN_MAX) * 2.5
            if depress_val > 0:
                p_eq -= depress_val / self.FLOW_IN_MAX * 1.5
                p_eq = max(p_eq, self.HEADER_P_OP_MIN)
            s["header_pressure"] += p_alpha * (p_eq - s["header_pressure"])
            s["header_pressure"] = np.clip(
                s["header_pressure"], 0.0, self.HEADER_P_DESIGN * 1.15
            )

            s["header_temperature"] = np.clip(
                s["header_temperature"]
                + self.np_random.normal(0.0, 0.3 / np.sqrt(self.SUBSTEPS)),
                self.TEMP_DESIGN_MIN,
                self.TEMP_DESIGN_MAX,
            )

            dilution = (
                flared / (flared + steam + 1e-8) if (flared + steam) > 0.0 else 1.0
            )
            nhv_cz = s["nhv_vg"] * dilution

            if flared > 10.0 and s["pilot_flame"] == 1.0:
                ce_base = 1.0 / (1.0 + np.exp(-0.015 * (nhv_cz - self.NHV_CZ_MIN)))
                sgr = steam / (flared + 1e-8)
                sgr_factor = 1.0 - 0.4 * min(abs(sgr - 0.3), 1.0)
                ce = float(
                    np.clip(
                        ce_base
                        * np.clip(sgr_factor, 0.7, 1.0)
                        * (1.0 + self.np_random.normal(0.0, 0.01)),
                        0.0,
                        1.0,
                    )
                )
                if tip_overload_ratio > 1.0:
                    liftoff_penalty = np.clip(
                        (tip_overload_ratio - 1.0) * 2.0, 0.0, 0.8
                    )
                    ce *= 1.0 - liftoff_penalty
            elif flared > 10.0:
                ce = 0.0
            else:
                ce = 1.0

            temp_factor = max(
                0.001,
                (self.TEMP_DESIGN_MAX - s["header_temperature"])
                / (self.TEMP_DESIGN_MAX - self.TEMP_DESIGN_MIN),
            )
            condensation = flow_in * 0.003 * temp_factor

            level_delta = (
                (condensation - ko_drain)
                * dt_sub_h
                / (self.KOD_VOLUME_M3 * self.KOD_LIQUID_DENSITY)
            )
            s["ko_drum_level"] = np.clip(s["ko_drum_level"] + level_delta, 0.0, 1.0)

            s["seal_drum_level"] = np.clip(
                s["seal_drum_level"]
                + (-0.001 + (0.0012 if s["seal_drum_level"] < 0.5 else 0.0))
                / self.SUBSTEPS,
                0.0,
                1.0,
            )

            noise = 35.0
            if flared > 0.0:
                noise = (
                    45.0 + 10.0 * np.log10(flared + 1.0) + 5.0 * np.log10(steam + 1.0)
                )
                if tip_overload_ratio > 1.0:
                    noise += 10.0 * np.log10(tip_overload_ratio)
            s["noise_db"] = np.clip(noise, 30.0, 160.0)

            # Actualizar estado termodinámico exacto
            s["flow_in"] = flow_in
            s["flared_kgh"] = flared
            s["steam_flow"] = steam
            s["purge_gas_flow"] = purge
            s["ko_drain_pv"] = ko_drain
            s["depress_pv"] = depress_val
            s["pcv_opening"] = float(pcv_opening)
            s["pcv_flow_kgh"] = float(pcv_flow)
            s["psv_open"] = bool(psv_open)
            s["psv_flow_kgh"] = float(psv_flow)
            s["nhv_cz"] = nhv_cz
            s["combustion_efficiency"] = ce

            h2s_emit_rate = s["h2s_in_gas"] * (1.0 - ce) * flared / 1000.0

            flared_int += flared * dt_sub_h
            steam_int += steam * dt_sub_h
            purge_int += purge * dt_sub_h
            pcv_flow_int += pcv_flow * dt_sub_h
            psv_flow_int += psv_flow * dt_sub_h
            ch4_fug_int += (1.0 - ce) * flared * 0.896 * dt_sub_h
            h2s_emit_int += h2s_emit_rate * dt_sub_h
            ce_acc += ce
            min_purge = min(min_purge, purge)

        dt_h = self.TIMESTEP_MIN / 60.0
        flared = flared_int / dt_h
        steam = steam_int / dt_h
        purge = purge_int / dt_h
        pcv_flow_avg = pcv_flow_int / dt_h
        psv_flow_avg = psv_flow_int / dt_h
        ce = ce_acc / self.SUBSTEPS
        h2s_emitted = h2s_emit_int / dt_h
        co2eq = ch4_fug_int * 80.0

        backpressure_ratio = min(
            s["header_pressure"] / (self.BACKPRESSURE_MAX_BARG * 10.197),
            1.5,
        )
        hydrate_risk = 0.0
        if s["header_temperature"] < 10.0 and s["header_pressure"] > 1.5:
            hydrate_risk = 1.0 - s["header_temperature"] / 10.0

        # ─── ESD ───────────────────────────────────────────
        esd = False
        esd_reason = ""
        if s["header_pressure"] > self.HEADER_P_DESIGN:
            esd, esd_reason = True, "header_overpressure"
        if min_purge < self.PURGE_GAS_MIN:
            esd, esd_reason = True, "purge_insufficient"
        if s["seal_drum_level"] < 0.15:
            esd, esd_reason = True, "seal_drum_low"
        if s["seal_drum_level"] > 0.85:
            esd, esd_reason = True, "seal_drum_high"
        if s["ko_drum_level"] > 0.85:
            esd, esd_reason = True, "ko_drum_overflow"
        s["esd_triggered"] = esd

        # ─── REWARD REFACTORIZADO ──────────────────────────
        # 1. Supervivencia y Estabilidad (Prioridad 1)
        r_survival = 2.0  # Bono por cada step vivo

        # Campo de Potencial para la Purga (Prevenir ESD por Purga)
        purge_buffer = self.PURGE_GAS_MIN * 1.25  # Alerta al 125% del mínimo
        r_safety_purge = 0.0
        if min_purge < purge_buffer:
            # Castigo cuadrático que crece al acercarse a los 30 kg/h
            r_safety_purge = (
                -8.0 * ((purge_buffer - min_purge) / self.PURGE_GAS_MIN) ** 2
            )

        # 2. Objetivos Operacionales y Alarmas (Prioridad 2)
        r_alarm = 0.0
        if s["header_pressure"] > self.HEADER_P_OP_MAX:
            r_alarm -= 3.0
        if backpressure_ratio > 0.8:
            r_alarm -= 3.0 * (backpressure_ratio - 0.8) / 0.2
        if s["noise_db"] > self.NOISE_EMERGENCY_DBA:
            r_alarm -= 5.0
        if s["nhv_cz"] < self.NHV_CZ_MIN and flared > 10.0:
            r_alarm -= 10.0
        if flare_overloaded:
            r_alarm -= 15.0

        # 3. Optimización Ambiental (Prioridad 3 - "Salvar el Planeta")
        r_flaring = -0.08 * flared_int
        r_steam = -self.STEAM_COST_PER_KG * steam_int
        r_incomplete = -(self.CO2_COST_PER_TON / 1000.0) * co2eq
        r_h2s = -1000.0 * h2s_emit_int

        # 3. Penalización por Eficiencia de Combustión (CE)
        r_ce_bonus = 0.0
        if flared > 10.0:
            if ce > 0.94:
                r_ce_bonus = 500.0 * (ce - 0.94)  # Incentivo lineal para subir
                # Bono extra por cumplimiento de ley (objetivo operacional)
                if ce >= self.CE_TARGET:
                    r_ce_bonus += 20.0
            else:
                r_ce_bonus = -200.0 * (0.94 - ce)  # Castigo por caer de donde ya está

        # Acotamos el impacto ambiental para que no "ahogue" el aprendizaje de supervivencia
        r_ambiental = max(-20.0, r_h2s + r_ce_bonus + r_incomplete + r_flaring)

        # 4. Estabilidad de Control
        delta = a - self.prev_action
        r_smooth = -0.2 * float(np.sum(delta**2))

        # 5. Castigo por ESD Dinámico (Dolerá más si mueres rápido)
        r_esd = 0.0
        if esd:
            steps_left = self.max_steps - self.steps
            r_esd = -(steps_left * r_survival) - 500.0

        reward = (
            r_survival
            + r_safety_purge
            + r_alarm
            + r_ambiental
            + r_steam
            + r_smooth
            + r_esd
        )

        self.steps += 1
        self.prev_action = a.copy()

        # ─── Métricas del episodio ─────────────────────────
        self.episode_metrics["total_flared_kg"] += flared_int
        self.episode_metrics["total_h2s_emitted"] += h2s_emit_int
        self.episode_metrics["total_co2eq_kg"] += co2eq / 1000.0
        self.episode_metrics["mean_ce"].append(ce)
        if esd:
            self.episode_metrics["safety_violations"] += 1
        if flare_overloaded:
            self.episode_metrics["flare_overload_steps"] += 1

        errors_norm = self._compute_error_norm()

        info = self._get_info()
        info.update(
            {
                "flow_in_kgh": s["flow_in"],
                "flared_kgh": flared,
                "recovered_kgh": 0.0,
                "recovery_pct": 0.0,
                "ce": ce,
                "flared_kgh": flared,
                "esd_triggered": esd,
                "esd_reason": esd_reason,
                "nhv_cz": s["nhv_cz"],
                "nhv_vg": s["nhv_vg"],
                "header_pressure_barg": s["header_pressure"],
                "header_temp_C": s["header_temperature"],
                "h2s_emitted": h2s_emitted,
                "noise_db": s["noise_db"],
                "hydrate_risk": hydrate_risk,
                "backpressure_ratio": backpressure_ratio,
                "flare_overloaded": flare_overloaded,
                "tip_capacity_pct": flared / (self.FLARE_CAPACITY_KGH + 1e-8) * 100.0,
                "esd_triggered": esd,
                "esd_reason": esd_reason,
                "upstream_p": s["upstream_p"],
                "pcv_setpoint": float(s["pcv_setpoint_pv"]),
                "pcv_flow_kgh": float(pcv_flow_avg),
                "pcv_opening_pct": float(s.get("pcv_opening", 0.0) * 100.0),
                "psv_open": bool(s.get("psv_open", False)),
                "psv_event": bool(s.get("psv_open", False)),
                "psv_flow_kgh": float(psv_flow_avg),
                "depress_flow_kgh": float(s["depress_pv"]),
                "controller": {
                    "setpoints": {
                        "steam_kgh": float(s["controller_sp"][0]),
                        "purge_kgh": float(s["controller_sp"][1]),
                        "kod_level_sp": float(s["controller_sp"][2]),
                        "pcv_setpoint": float(s["pcv_setpoint_pv"]),
                    },
                    "pvs": {
                        "steam_kgh": float(s["steam_flow"]),
                        "purge_kgh": float(s["purge_gas_flow"]),
                        "kod_level": float(s["ko_drum_level"]),
                        "header_pressure": float(s["header_pressure"]),
                        "upstream_p": float(s["upstream_p"]),
                        "kod_drain_pv": float(s["ko_drain_pv"]),
                        "depress_pv": float(s["depress_pv"]),
                        "pcv_setpoint": float(s["pcv_setpoint_pv"]),
                    },
                    "inner_setpoints": {
                        "kod_drain_sp": float(s["inner_sp"][0]),
                        "depress_sp": float(s["inner_sp"][1]),
                    },
                    "errors_norm": errors_norm.tolist(),
                },
                "episode_metrics": {
                    **self.episode_metrics,
                    "mean_ce": float(np.mean(self.episode_metrics["mean_ce"]))
                    if self.episode_metrics["mean_ce"]
                    else 0.0,
                },
                "reward_components": {
                    "flaring": r_flaring,
                    "incomplete": r_incomplete,
                    "steam": r_steam,
                    "smoothness": r_smooth,
                    "esd": r_esd,
                    "survival": r_survival,
                    "safety_purge": r_safety_purge,
                    "ambiental": r_ambiental,
                    "esd_penalty": r_esd,
                    "alarms": r_alarm,
                },
            }
        )

        terminated = esd
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        s = self.state
        err_norm = self._compute_error_norm()

        process = np.array(
            [
                s["flow_in"] / self.FLOW_IN_MAX,
                s["nhv_vg"] / 1500.0,
                s["header_pressure"] / (self.HEADER_P_DESIGN * 1.15),
                (s["header_temperature"] - self.TEMP_DESIGN_MIN)
                / (self.TEMP_DESIGN_MAX - self.TEMP_DESIGN_MIN),
                s["nhv_cz"] / 1500.0,
                s["combustion_efficiency"],
                s["seal_drum_level"],
                s["ko_drum_level"],
                s["purge_gas_flow"] / self.PURGE_GAS_MAX,
                s["steam_flow"] / self.STEAM_MAX,
                s["pilot_flame"],
                s["h2s_in_gas"] / self.H2S_MAX_MGNM3,
                s["noise_db"] / self.NOISE_MAX_IMPULSIVE,
                (s["upstream_p"] - self.UPSTREAM_P_MIN)
                / (self.UPSTREAM_P_PSV_SET - self.UPSTREAM_P_MIN),
                s["steam_flow"]
                / (s["flared_kgh"] + 1e-8)
                / 0.5,  # Termodinámica exacta SGR
                self.steps / self.max_steps,
            ],
            dtype=np.float32,
        )
        process = np.clip(process, 0.0, 1.0)
        return np.concatenate([process, err_norm]).astype(np.float32)

    def _get_info(self):
        return {
            "step": self.steps,
            "time_hours": self.steps * self.TIMESTEP_MIN / 60.0,
        }


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = FlareSystemEnv()
    print("=" * 70)
    print("SimPlant — FlareSystemEnv SIN FGRS (calibrado)")
    print("=" * 70)
    print(f"Obs: {env.observation_space.shape} | Act: {env.action_space.shape}")
    print()
    print("── Equipos calibrados ──")
    print(
        f"  TRKO-261 KOD:  D={env.KOD_DIAMETER_M}m × L={env.KOD_LENGTH_M}m"
        f" → V={env.KOD_VOLUME_M3:.1f} m³"
    )
    print(
        f"  B-262/63:      {env.KOD_PUMP_COUNT}× centrífugas,"
        f" {env.KOD_PUMP_FLOW_M3H} m³/h, {env.KOD_PUMP_POWER_HP} HP"
    )
    print(
        f'  L-264 Flare:   {env.FLARE_TIP_DIAMETER_IN}",'
        f" {env.FLARE_STACK_HEIGHT_M}m,"
        f" {env.FLARE_PILOT_COUNT} pilotos,"
        f" {env.FLARE_CAPACITY_SM3D / 1e6:.0f} MM Sm³/d"
        f" → {env.FLARE_CAPACITY_KGH:.0f} kg/h"
    )
    print()
    print(
        f"  PCV Cv={env.PCV_CV:.0f} |"
        f" SP=[{env.PCV_SP_MIN:.0f}, {env.PCV_SP_MAX:.0f}] kg/cm²-g"
    )
    print(
        f"  PSV set={env.UPSTREAM_P_PSV_SET:.0f} kg/cm²-g |"
        f" pico={env.PSV_PEAK_FLOW:.0f} kg/h"
    )
    print(
        f"  Depress: activa > {env.DEPRESS_ENABLE_P} barg |"
        f" max={env.DEPRESS_MAX_FLOW_KGH:.0f} kg/h"
    )
    print()

    check_env(env.unwrapped, skip_render_check=True)
    print("Gymnasium check_env: OK")
    print()

    for name, policy in [
        ("RANDOM", lambda _: env.action_space.sample()),
        ("HOLD_ZERO", lambda _: np.zeros(3, dtype=np.float32)),
    ]:
        obs, _ = env.reset(seed=42)
        total_r = 0.0
        for _ in range(48):
            obs, r, term, trunc, info = env.step(policy(obs))
            total_r += r
            if term or trunc:
                break
        metrics = info["episode_metrics"]
        print(
            f"{name:10s} | t={info['time_hours']:.1f}h | R={total_r:+.1f} | "
            f"Flared={metrics['total_flared_kg']:.1f}kg | "
            f"Tip%={info.get('tip_capacity_pct', 0):.1f}% | "
            f"ESD={info['esd_triggered']}"
        )

