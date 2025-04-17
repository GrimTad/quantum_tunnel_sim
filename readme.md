# 量子トンネル効果シミュレーター / Quantum Tunneling Simulator

## 日本語説明

### 概要
このプログラムは一次元シュレディンガー方程式を数値的に解くことで、量子力学における重要な現象である「量子トンネル効果」をシミュレーションします。粒子の波動性により、古典力学では越えられないエネルギー障壁をすり抜ける現象を視覚的に理解することができます。

### 特徴
- Crank-Nicolson法による時間発展の高精度計算
- 様々なポテンシャル形状（障壁、井戸、二重井戸など）のシミュレーション
- リアルタイムでパラメータを調整できるインタラクティブモード
- エネルギーと透過率の関係を分析する機能
- 詳細な物理量の表示と教育的な解説テキスト

### 必要条件
- Python 3.6以上
- NumPy
- SciPy
- Matplotlib
- ffmpeg（アニメーション保存時のみ）

### インストール方法
```bash
pip install numpy scipy matplotlib
# アニメーション保存機能を使用する場合
# ffmpegをインストール（OSによりインストール方法が異なります）
```

### 使用方法

#### 基本実行
```bash
python quantum_tunnel_sim.py
```
デフォルトの設定で量子トンネル効果のシミュレーションを実行し、結果をグラフで表示します。

#### コマンドラインオプション
```bash
# インタラクティブモード
python quantum_tunnel_sim.py --mode interactive

# 異なるポテンシャルの比較
python quantum_tunnel_sim.py --mode potentials

# エネルギーと透過率の関係分析
python quantum_tunnel_sim.py --mode energy

# アニメーションをMP4ファイルとして保存（ffmpegが必要）
python quantum_tunnel_sim.py --save

# 位相プロットを無効化
python quantum_tunnel_sim.py --no-phase
```

### モードの説明

1. **基本モード (main)**  
   デフォルトの設定で量子トンネル効果をシミュレーションし、波動関数の時間発展を表示します。

2. **インタラクティブモード (interactive)**  
   スライダーを使って波動関数のパラメータ（波数、幅）やポテンシャルの特性（高さ、幅）をリアルタイムで調整できます。シミュレーション結果への影響を直感的に理解するのに最適です。

3. **ポテンシャル比較モード (potentials)**  
   バリア、井戸、二重井戸など異なるタイプのポテンシャルでの波動関数の振る舞いを比較します。

4. **エネルギー分析モード (energy)**  
   波動関数のエネルギーとバリア透過率の関係を分析し、グラフ化します。量子トンネル効果の基本的な特性を理解するのに役立ちます。
### インタラクティブモードの使い方
1. `--mode interactive`オプションでプログラムを起動します
2. 以下のスライダーを使ってパラメータを調整できます：
   - **Time t**: 時間発展のタイムステップを変更
   - **Wave Number k₀**: 波動関数の波数（運動量/エネルギーに関連）
   - **Wave Width σ**: 波束の幅（位置の不確定性）
   - **Potential Height V₀**: ポテンシャル障壁の高さ
   - **Barrier Width**: ポテンシャル障壁の幅
3. **Reset**ボタンで初期状態に戻せます

### 物理的背景
量子トンネル効果は、粒子が自身のエネルギーより高いポテンシャル障壁を透過する現象です。古典力学では不可能なこの現象は、量子力学の波動性から生じます。

主要なポイント：
- 波束のエネルギーがバリアより低いほど、透過率は指数関数的に減少します
- バリアが薄いほど、透過率は高くなります
- このシミュレーションでは、シュレディンガー方程式の数値解を用いて波動関数の時間発展を計算しています

### コードの解説
このシミュレーターは、以下の主要なコンポーネントで構成されています：

1. **QuantumTunnelSimulator クラス**: シミュレーションの中核となるクラスで、以下の機能を提供：
   - シュレディンガー方程式の数値解法（Crank-Nicolson法）
   - 波動関数の初期化と時間発展
   - 様々なポテンシャル形状の生成
   - 物理量（エネルギー、位置期待値など）の計算

2. **視覚化コンポーネント**:
   - 静的プロット、アニメーション、インタラクティブなGUIを提供
   - 複数のグラフを使った結果の表示
   - リアルタイムでのパラメータ調整

3. **解析モード**:
   - 異なるポテンシャルでの波動関数の比較
   - エネルギーと透過率の関係の解析

コード内の計算方法の概要：
```python
# シュレディンガー方程式の時間発展（Crank-Nicolson法）
# (I + H*dt/(2i*hbar))*psi(t+dt) = (I - H*dt/(2i*hbar))*psi(t)
# これを行列方程式として解く
b = self.B @ psi  # 右辺の計算
psi = spsolve(self.A, b)  # 線形方程式を解いて次の時間ステップの波動関数を得る
```

## English Description

### Overview
This program simulates the quantum tunneling effect by numerically solving the one-dimensional Schrödinger equation. It provides a visual understanding of how a particle, due to its wave nature, can penetrate energy barriers that would be impossible to overcome in classical mechanics.

### Features
- High-precision time evolution using the Crank-Nicolson method
- Simulation of various potential shapes (barrier, well, double well, etc.)
- Interactive mode to adjust parameters in real-time
- Analysis of the relationship between energy and transmission probability
- Detailed display of physical quantities and educational explanatory text

### Requirements
- Python 3.6 or higher
- NumPy
- SciPy
- Matplotlib
- ffmpeg (only for saving animations)

### Installation
```bash
pip install numpy scipy matplotlib
# If you want to save animations
# Install ffmpeg (installation method varies by OS)
```

### Usage

#### Basic Execution
```bash
python quantum_tunnel_sim.py
```
This runs the quantum tunneling simulation with default settings and displays the results graphically.

#### Command Line Options
```bash
# Interactive mode
python quantum_tunnel_sim.py --mode interactive

# Comparison of different potentials
python quantum_tunnel_sim.py --mode potentials

# Analysis of energy vs. transmission relationship
python quantum_tunnel_sim.py --mode energy

# Save animation as MP4 file (requires ffmpeg)
python quantum_tunnel_sim.py --save

# Disable phase plot
python quantum_tunnel_sim.py --no-phase
```

### Mode Descriptions

1. **Basic Mode (main)**  
   Simulates quantum tunneling with default settings and displays the time evolution of the wave function.

2. **Interactive Mode (interactive)**  
   Allows real-time adjustment of wave function parameters (wave number, width) and potential characteristics (height, width) using sliders. Ideal for intuitively understanding how changes affect the simulation results.

3. **Potential Comparison Mode (potentials)**  
   Compares the behavior of wave functions in different types of potentials such as barriers, wells, and double wells.

4. **Energy Analysis Mode (energy)**  
   Analyzes and graphs the relationship between the wave function's energy and barrier transmission probability. Useful for understanding the basic characteristics of quantum tunneling.

### How to Use Interactive Mode
1. Start the program with the `--mode interactive` option
2. Adjust parameters using the following sliders:
   - **Time t**: Change the time evolution time step
   - **Wave Number k₀**: Wave function's wave number (related to momentum/energy)
   - **Wave Width σ**: Width of the wave packet (position uncertainty)
   - **Potential Height V₀**: Height of the potential barrier
   - **Barrier Width**: Width of the potential barrier
3. Use the **Reset** button to return to the initial state

### Physical Background
Quantum tunneling is a phenomenon where particles can penetrate potential barriers higher than their own energy. This phenomenon, impossible in classical mechanics, arises from the wave nature of quantum mechanics.

Key points:
- The lower the wave packet's energy compared to the barrier, the more exponentially the transmission probability decreases
- The thinner the barrier, the higher the transmission probability
- This simulation calculates the time evolution of the wave function using a numerical solution of the Schrödinger equation

### Code Explanation
This simulator consists of the following main components:

1. **QuantumTunnelSimulator Class**: The core class of the simulation, providing:
   - Numerical solution of the Schrödinger equation (Crank-Nicolson method)
   - Initialization and time evolution of the wave function
   - Generation of various potential shapes
   - Calculation of physical quantities (energy, position expectation values, etc.)

2. **Visualization Components**:
   - Static plots, animations, and interactive GUI
   - Display of results using multiple graphs
   - Real-time parameter adjustment

3. **Analysis Modes**:
   - Comparison of wave functions in different potentials
   - Analysis of the relationship between energy and transmission probability

Overview of the calculation method in the code:
```python
# Time evolution of the Schrödinger equation (Crank-Nicolson method)
# (I + H*dt/(2i*hbar))*psi(t+dt) = (I - H*dt/(2i*hbar))*psi(t)
# Solved as a matrix equation
b = self.B @ psi  # Calculate right-hand side
psi = spsolve(self.A, b)  # Solve linear equation to get wave function at next time step
``` 
