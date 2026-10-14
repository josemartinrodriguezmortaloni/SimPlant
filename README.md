# SimPlant Trainer V2 🌱🔥

¡Hola! Bienvenido/a a **SimPlant Trainer**. 

Este documento está armado paso a paso para que entiendas a la perfección qué hace este proyecto y cómo podés usarlo sin volverte loco/a.

## ¿De qué se trata esto? 🤔

En una planta industrial, la **antorcha (flare)** es un equipo crítico de seguridad que quema los excesos de gas, pero esto genera emisiones al medio ambiente y pérdida de producto. 

Este proyecto utiliza **Inteligencia Artificial (IA)** —específicamente una técnica que se llama Aprendizaje por Refuerzo — para entrenar un "cerebro virtual" llamado Soft Actor-Critic (SAC). El objetivo de esta IA es aprender a controlar la antorcha de la manera más óptima posible para:
1. **Recuperar gas** que de otra forma se quemaría (beneficio económico).
2. **Reducir las emisiones** al medio ambiente (beneficio ecológico).
3. **Mantener la seguridad operativa** de la planta al 100% (esto es innegociable).

Básicamente, nosotros le damos a la IA un simulador de la planta y la dejamos probar miles de veces (equivocándose y aprendiendo) hasta que descubre la mejor forma de operar.

---

## Estructura: ¿Qué hay en estas carpetas? 📂

Para que te ubiques rápido y fácil, el proyecto está dividido así:

- `src/simplant_env/`: Acá viven los "simuladores" de la planta. Es el mundo virtual donde la IA practica.
- `src/simplant_trainer/`: Acá está el código del "cerebro" (la IA) que va aprendiendo a tomar decisiones.
- `models/`: Acá están los scripts (los archivos ejecutables) que vos vas a correr para decirle a la computadora: *"¡Empezá a entrenar!"*.
- `results/`: ¡Acá está lo que nos importa! Una vez que la IA termina de entrenar, te va a dejar gráficos y reportes de desempeño en esta carpeta.

---

## 🛠️ Paso 1: Preparar la computadora (Instalación)

Para que esto funcione, necesitamos instalar un par de herramientas base. Es súper directo, seguí estos pasos:

### 1. El requisito fundamental
Necesitás tener instalado **Python** (versión 3.13 o superior) en tu compu. Si no lo tenés, podés descargarlo e instalarlo fácil desde [python.org](https://www.python.org/downloads/).

### 2. Instalar el gestor de paquetes (`uv`)
Para no renegar instalando cosas a mano, usamos una herramienta buenísima y rapidísima que se llama `uv`. 
Abrí tu terminal o consola (Command Prompt o PowerShell en Windows) y pegá este comando para instalarlo:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Instalar el proyecto
Una vez que tenés `uv` instalado, asegurate de estar parado en la carpeta de este proyecto (`SimPlant-Trainer`) con tu terminal y ejecutá este comando:

```bash
uv sync
```
¡Listo! Con eso, la computadora ya descargó y configuró todo lo necesario de forma automática.

*(Nota para técnicos: Si preferís la vieja escuela, podés crear un entorno virtual con `python -m venv .venv`, activarlo e instalar con `pip install -e .`)*

---

## 🚀 Paso 2: Poner a entrenar a la Inteligencia Artificial

Tenemos dos escenarios de simulación preparados. Elegí el que necesites probar y ejecutá el comando correspondiente en tu terminal.

### Opción A: Entrenamiento con Sistema de Recuperación de Gases (FGRS)
Este escenario simula una planta que **SÍ** tiene un equipo para recuperar el gas de la antorcha. Es el escenario más completo e ideal.

**Para arrancarlo, ejecutá:**
```bash
uv run models/train_env.py
```
*Cuando termine, vas a encontrar los resultados guardados en la carpeta `results/results_flare_env1/`.*

### Opción B: Entrenamiento sin Recuperación de Gases (Sin FGRS)
Este escenario simula una planta tradicional que **no** tiene sistema de recuperación. La IA solo se enfoca en quemar el gas de la forma más limpia y eficiente posible.

**Para arrancarlo, ejecutá:**
```bash
uv run models/train_env2.py
```
*Cuando termine, los resultados se van a guardar en la carpeta `results/results_flare_env2/`.*

> **💡 Un tip:** Los entrenamientos simulan muchísimas horas de operación de la planta y pueden tardar. Si ves que demora mucho y querés hacer una prueba rápida, podés abrir los archivos `.py` de la carpeta `models` y achicar los números donde dice `NUM_EPOCHS` o `STEPS_PER_EPOCH`.

---

## 📊 Paso 3: ¿Cómo leo los resultados?

Una vez que el entrenamiento termina (o incluso mientras está corriendo), el sistema te va a generar archivos para que veas qué tan bien aprendió la IA. Vas a encontrar todo dentro de la carpeta `results/`.

Acá te explico qué es cada cosa, en lenguaje claro:

### 1. El Reporte Escrito (`report.txt`)
**Este es el archivo principal para que tomes decisiones.**
Es un resumen ejecutivo que te cuenta cómo le fue a la IA comparado con un operador humano de perfil conservador (valores estimados).
Fijate en la matriz de cumplimiento al final del archivo:
- `✅ CUMPLE`: La IA logró operar la planta por encima de los objetivos requeridos.
- `⚠️ BORDE`: La IA está al límite operativo, hay que prestar atención.
- `❌ NO CUMPLE`: Acá la IA cometió un error o desvío (por ejemplo, levantó mucha presión).

*Consejo: Mirá siempre primero la Seguridad de la planta, luego el Cumplimiento ambiental, y por último el Beneficio económico (gas recuperado).*

### 2. Comparación Operativa (`sac_vs_conservative.png`)
Este gráfico te muestra "un día entero en la planta" (24 horas).
Vas a ver una línea para la IA compitiendo contra la línea de un control tradicional. Mirá especialmente los gráficos de **"Eficiencia de Combustión (CE)"** y el **flujo de gas recuperado**. Si la línea de la IA va por arriba en eficiencia y recupera más gas sin activar alarmas de emergencia, ¡es un éxito rotundo!

### 3. Curvas de Aprendizaje (`training_curves.png`)
Este gráfico es un poquito más técnico. Te muestra cómo fue el proceso de aprendizaje de la IA a lo largo del tiempo. Si ves que la curva de "Reward" (recompensa o puntaje) sube y luego se estabiliza, significa que la IA entendió las reglas del juego y encontró una estrategia operativa ganadora.

### 4. El "Cerebro" Guardado (`.safetensors`)
Vas a ver unos archivos con esta extensión rara. Son literalmente los "pesos neuronales" (la memoria de lo aprendido) de la IA. Si el día de mañana queremos conectar esta Inteligencia Artificial a la planta real o a un software más grande, usamos este archivo. ¡No lo borres!