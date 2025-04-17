import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

class QuantumTunnelSimulator:
    """
    Numerical solution of one-dimensional Schrödinger equation using the Crank-Nicolson method
    for simulating quantum tunneling effect
    """
    
    def __init__(self, config=None):
        """
        Initialize the simulation
        
        Parameters:
        -----------
        config : dict, optional
            Dictionary containing simulation parameters
        """
        # Default settings
        self.default_config = {
            # Spatial grid settings
            'x_min': -10.0,       # Minimum x value
            'x_max': 10.0,        # Maximum x value
            'nx': 1000,           # Number of spatial grid points
            
            # Time evolution settings
            'dt': 0.005,          # Time step
            'nt': 1000,           # Number of time steps
            'save_every': 5,      # Save results every n steps
            
            # Initial wavefunction settings
            'x0': -5.0,           # Initial position
            'sigma': 0.5,         # Width of Gaussian packet
            'k0': 5.0,            # Initial wave number (momentum)
            
            # Potential settings
            'potential_type': 'barrier',  # Type of potential: 'barrier', 'well', 'double_well', 'custom'
            'V0': 1.0,            # Potential height/depth
            'barrier_width': 0.5, # Barrier width
            'barrier_pos': 0.0,   # Barrier position
            
            # Physical constants
            'hbar': 1.0,          # Reduced Planck constant
            'm': 1.0,             # Particle mass
        }
        
        # Override with user settings
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Extract settings
        self.x_min = self.config['x_min']
        self.x_max = self.config['x_max']
        self.nx = self.config['nx']
        self.dt = self.config['dt']
        self.nt = self.config['nt']
        self.save_every = self.config['save_every']
        self.x0 = self.config['x0']
        self.sigma = self.config['sigma']
        self.k0 = self.config['k0']
        self.potential_type = self.config['potential_type']
        self.V0 = self.config['V0']
        self.barrier_width = self.config['barrier_width']
        self.barrier_pos = self.config['barrier_pos']
        self.hbar = self.config['hbar']
        self.m = self.config['m']
        
        # Create spatial grid
        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        
        # Crank-Nicolson method parameter
        self.alpha = 1j * self.hbar * self.dt / (4 * self.m * self.dx**2)
        
        # Initialize potential
        self.V = self._initialize_potential()
        
        # Initialize wavefunction
        self.psi_init = self._initialize_wavefunction()
        self.normalize_wavefunction(self.psi_init)
        
        # Prepare time evolution operators
        self._prepare_time_evolution_operators()
        
        # Arrays to store results
        self.n_saves = self.nt // self.save_every + 1
        self.psi_history = np.zeros((self.n_saves, self.nx), dtype=complex)
        self.t_history = np.zeros(self.n_saves)
        
        # Analysis results
        self.transmission = 0.0
        self.reflection = 0.0
    
    def _initialize_potential(self):
        """Initialize the potential"""
        V = np.zeros(self.nx)
        
        if self.potential_type == 'barrier':
            # Barrier potential
            barrier_start = np.abs(self.x - (self.barrier_pos - self.barrier_width/2)).argmin()
            barrier_end = np.abs(self.x - (self.barrier_pos + self.barrier_width/2)).argmin()
            V[barrier_start:barrier_end+1] = self.V0
        
        elif self.potential_type == 'well':
            # Well potential
            well_start = np.abs(self.x - (self.barrier_pos - self.barrier_width/2)).argmin()
            well_end = np.abs(self.x - (self.barrier_pos + self.barrier_width/2)).argmin()
            V[:] = self.V0
            V[well_start:well_end+1] = 0.0
        
        elif self.potential_type == 'double_well':
            # Double well potential
            well_width = self.barrier_width / 3
            gap_width = self.barrier_width / 3
            
            left_well_start = np.abs(self.x - (self.barrier_pos - self.barrier_width/2)).argmin()
            left_well_end = np.abs(self.x - (self.barrier_pos - self.barrier_width/2 + well_width)).argmin()
            
            right_well_start = np.abs(self.x - (self.barrier_pos + self.barrier_width/2 - well_width)).argmin()
            right_well_end = np.abs(self.x - (self.barrier_pos + self.barrier_width/2)).argmin()
            
            V[:] = self.V0
            V[left_well_start:left_well_end+1] = 0.0
            V[right_well_start:right_well_end+1] = 0.0
        
        elif self.potential_type == 'custom':
            # Custom potential - implement as needed
            pass
        
        return V
    
    def _initialize_wavefunction(self):
        """Initialize the wavefunction (Gaussian packet)"""
        # Gaussian packet: psi(x) = exp(-(x-x0)^2/(2*sigma^2)) * exp(i*k0*x)
        psi = np.exp(-((self.x - self.x0)**2) / (2 * self.sigma**2)) * np.exp(1j * self.k0 * self.x)
        return psi
    
    def normalize_wavefunction(self, psi):
        """Normalize wavefunction"""
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        if norm > 0:
            psi /= norm
        return psi
    
    def _prepare_time_evolution_operators(self):
        """Prepare time evolution operators for the Crank-Nicolson method"""
        # Diagonal elements
        diag_elements = 1 + 2 * self.alpha + 1j * self.dt * self.V / (2 * self.hbar)
        
        # Off-diagonal elements
        off_diag_elements = -self.alpha * np.ones(self.nx-1)
        
        # Left-hand side operator (I + H*dt/(2i*hbar))
        self.A = diags(
            [off_diag_elements, diag_elements, off_diag_elements],
            [-1, 0, 1],
            shape=(self.nx, self.nx),
            dtype=complex
        ).tocsc()  # Convert to CSC format to avoid warning
        
        # Right-hand side operator (I - H*dt/(2i*hbar))
        diag_elements_B = 1 - 2 * self.alpha - 1j * self.dt * self.V / (2 * self.hbar)
        self.B = diags(
            [self.alpha * np.ones(self.nx-1), diag_elements_B, self.alpha * np.ones(self.nx-1)],
            [-1, 0, 1],
            shape=(self.nx, self.nx),
            dtype=complex
        )
    
    def time_evolve(self):
        """Calculate time evolution of wavefunction"""
        psi = self.psi_init.copy()
        
        # Save initial values
        self.psi_history[0] = psi
        self.t_history[0] = 0.0
        
        save_idx = 1
        
        for n in range(1, self.nt + 1):
            # Time evolution using Crank-Nicolson method
            # Solve A*psi(t+dt) = B*psi(t)
            b = self.B @ psi
            
            # Boundary conditions: wavefunction is zero at boundaries
            b[0] = 0
            b[-1] = 0
            
            # Solve the system of equations
            psi = spsolve(self.A, b)
            
            # Normalize wavefunction
            self.normalize_wavefunction(psi)
            
            # Save results when needed
            if n % self.save_every == 0:
                self.psi_history[save_idx] = psi
                self.t_history[save_idx] = n * self.dt
                save_idx += 1
        
        # Calculate transmission and reflection coefficients
        self.calculate_transmission_reflection()
        
        return self.psi_history, self.t_history
    
    def calculate_transmission_reflection(self):
        """Calculate transmission and reflection coefficients"""
        # Final wavefunction
        psi_final = self.psi_history[-1]
        
        # Divide into left side and right side using barrier position
        barrier_idx = np.abs(self.x - self.barrier_pos).argmin()
        
        # Calculate probabilities on left (reflection) and right (transmission) sides
        prob_left = np.sum(np.abs(psi_final[:barrier_idx])**2) * self.dx
        prob_right = np.sum(np.abs(psi_final[barrier_idx:])**2) * self.dx
        
        # Transmission and reflection coefficients
        self.transmission = prob_right
        self.reflection = prob_left
        
        return self.transmission, self.reflection
    
    def get_probability_density(self, psi=None):
        """Calculate probability density from wavefunction"""
        if psi is None:
            psi = self.psi_history[-1]
        return np.abs(psi)**2
    
    def get_phase(self, psi=None):
        """Calculate phase from wavefunction"""
        if psi is None:
            psi = self.psi_history[-1]
        return np.angle(psi)
    
    def get_expected_position(self, psi=None):
        """Calculate expected position value"""
        if psi is None:
            psi = self.psi_history[-1]
        prob = self.get_probability_density(psi)
        return np.sum(self.x * prob) * self.dx / np.sum(prob * self.dx)
    
    def get_energy(self, psi=None):
        """Calculate expected energy value"""
        if psi is None:
            psi = self.psi_history[-1]
            
        # Calculate kinetic energy expectation value
        # Approximate second derivative using central differences
        d2psi_dx2 = np.zeros_like(psi, dtype=complex)
        d2psi_dx2[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / self.dx**2
        
        # Use forward/backward differences for endpoints
        d2psi_dx2[0] = (psi[2] - 2*psi[1] + psi[0]) / self.dx**2
        d2psi_dx2[-1] = (psi[-1] - 2*psi[-2] + psi[-3]) / self.dx**2
        
        # Kinetic energy = -ħ²/(2m) * ∫ψ*(d²ψ/dx²)dx
        kinetic_energy = -self.hbar**2 / (2 * self.m) * np.sum(np.conj(psi) * d2psi_dx2).real * self.dx
        
        # Potential energy expectation value
        potential_energy = np.sum(self.V * self.get_probability_density(psi)) * self.dx
        
        return kinetic_energy + potential_energy
    
    def plot_results(self, frame=None, ax=None, fig=None, include_phase=False):
        """Plot results"""
        if ax is None:
            if include_phase:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax = ax1
                ax_phase = ax2
            else:
                fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.clear()
        
        # Use the last frame if none specified
        if frame is None:
            frame = -1
        
        # Plot probability density
        psi = self.psi_history[frame]
        prob = self.get_probability_density(psi)
        ax.plot(self.x, prob, 'b-', label='Probability Density |ψ|²')
        
        # Plot potential (with fixed scale)
        # 第二のY軸をポテンシャル用に作成
        ax_potential = ax.twinx()
        ax_potential.plot(self.x, self.V, 'r-', label='Potential V(x)')
        ax_potential.set_ylabel('Potential V(x)', color='r')
        ax_potential.tick_params(axis='y', labelcolor='r')
        
        # Mark expected position
        expected_x = self.get_expected_position(psi)
        ax.axvline(x=expected_x, color='g', linestyle='--', label='Expected Position <x>')
        
        # Plot settings
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(0, np.max(prob) * 1.1)
        ax_potential.set_ylim(0, np.max(self.V) * 1.1 if np.max(self.V) > 0 else 1.0)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Probability Density', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_title(f'Time t = {self.t_history[frame]:.3f}')
        
        # レジェンドを両方の軸に対応
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_potential.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot phase (optional)
        if include_phase:
            ax_phase.clear()
            phase = self.get_phase(psi)
            ax_phase.plot(self.x, phase, 'g-', label='Phase arg(ψ)')
            ax_phase.set_xlabel('Position x')
            ax_phase.set_ylabel('Phase (rad)')
            ax_phase.set_ylim(-np.pi, np.pi)
            ax_phase.legend(loc='upper right')
        
        return fig
    
    def create_animation(self, interval=50, include_phase=False):
        """Create animation with enhanced visual information"""
        if include_phase:
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
            ax = fig.add_subplot(gs[0])
            ax_phase = fig.add_subplot(gs[1], sharex=ax)
            ax_info = fig.add_subplot(gs[2])
            ax_info.axis('off')  # テキスト情報表示用
        else:
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            ax = fig.add_subplot(gs[0])
            ax_phase = None
            ax_info = fig.add_subplot(gs[1])
            ax_info.axis('off')  # テキスト情報表示用
        
        # ポテンシャル用の第二のY軸を作成
        ax_potential = ax.twinx()
        
        # 物理現象の説明テキスト
        tunneling_text = """
        量子トンネル効果: 古典力学では禁止されているエネルギー障壁の透過現象。
        粒子の波動性により、確率的に障壁を「すり抜ける」ことができる。
        
        ポイント:
        • 波束のエネルギーが障壁より低いほど、透過率は指数関数的に減少
        • 障壁が薄いほど、透過しやすい
        • シュレディンガー方程式の波動関数は虚数時間で拡散方程式と同形
        """
        
        # 数式表示用のテキスト
        equation_text = r"$i\hbar\frac{\partial\psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\psi}{\partial x^2} + V(x)\psi$"
        
        # 情報表示の初期化
        info_text = ax_info.text(0.5, 0.5, "", ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        def update(frame):
            ax.clear()
            ax_potential.clear()
            
            # Plot probability density
            psi = self.psi_history[frame]
            prob = self.get_probability_density(psi)
            ax.plot(self.x, prob, 'b-', label='確率密度 |ψ|²', linewidth=2)
            
            # Plot potential (with fixed scale)
            ax_potential.plot(self.x, self.V, 'r-', label='ポテンシャル V(x)', linewidth=2)
            ax_potential.set_ylabel('ポテンシャル V(x)', color='r', fontsize=10)
            ax_potential.tick_params(axis='y', labelcolor='r')
            
            # 粒子のエネルギーを計算（近似値）
            energy = self.get_energy(psi)
            kinetic = self.hbar**2 * self.k0**2 / (2 * self.m)
            
            # Mark expected position
            expected_x = self.get_expected_position(psi)
            ax.axvline(x=expected_x, color='g', linestyle='--', label='期待値 <x>', linewidth=1.5)
            
            # エネルギーレベルを示す水平線（ポテンシャル軸のスケールで）
            if np.max(self.V) > 0:
                energy_scaled = energy / np.max(self.V) * np.max(self.V)
                ax_potential.axhline(y=energy_scaled, color='purple', linestyle='-.', 
                                   label=f'エネルギー: {energy:.2f}', linewidth=1.5)
            
            # 透過率と反射率の計算
            barrier_idx = np.abs(self.x - self.barrier_pos).argmin()
            trans_prob = np.sum(np.abs(psi[barrier_idx:])**2) * self.dx
            refl_prob = np.sum(np.abs(psi[:barrier_idx])**2) * self.dx
            
            # バリアの位置をマーク
            barrier_start = np.abs(self.x - (self.barrier_pos - self.barrier_width/2)).argmin()
            barrier_end = np.abs(self.x - (self.barrier_pos + self.barrier_width/2)).argmin()
            ax.axvspan(self.x[barrier_start], self.x[barrier_end], alpha=0.2, color='red')
            
            # Plot settings
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(0, np.max(self.get_probability_density(self.psi_history[0])) * 1.1)
            ax_potential.set_ylim(0, np.max(self.V) * 1.1 if np.max(self.V) > 0 else 1.0)
            ax.set_xlabel('位置 x', fontsize=10)
            ax.set_ylabel('確率密度', color='b', fontsize=10)
            ax.tick_params(axis='y', labelcolor='b')
            ax.set_title(f'量子トンネル効果シミュレーション (時間 t = {self.t_history[frame]:.3f})', 
                       fontsize=12, fontweight='bold')
            
            # レジェンドを両方の軸に対応
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_potential.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
            
            # 情報テキストの更新
            info_string = f"""
            シミュレーションパラメータ:
            波数 k₀ = {self.k0:.2f}    波束幅 σ = {self.sigma:.2f}    位置期待値 <x> = {expected_x:.2f}
            ポテンシャル高さ V₀ = {self.V0:.2f}    バリア幅 = {self.barrier_width:.2f}
            
            解析結果:
            エネルギー = {energy:.4f}    運動エネルギー ≈ {kinetic:.4f}
            透過確率 = {trans_prob:.4f}    反射確率 = {refl_prob:.4f}
            
            {equation_text}
            
            {tunneling_text}
            """
            info_text.set_text(info_string)
            
            # Plot phase (optional)
            if include_phase and ax_phase:
                ax_phase.clear()
                phase = self.get_phase(psi)
                ax_phase.plot(self.x, phase, 'g-', label='位相 arg(ψ)', linewidth=2)
                
                # 位相の重要ポイントを強調
                phase_derivative = np.gradient(phase, self.x)
                ax_phase.plot(self.x, phase_derivative / np.max(np.abs(phase_derivative)) * np.pi * 0.5, 
                            'm--', label='位相勾配 (運動量に比例)', linewidth=1, alpha=0.7)
                
                ax_phase.set_xlabel('位置 x', fontsize=10)
                ax_phase.set_ylabel('位相 (rad)', fontsize=10)
                ax_phase.set_ylim(-np.pi, np.pi)
                ax_phase.set_title('波動関数の位相と位相勾配（運動量に比例）', fontsize=10)
                ax_phase.legend(loc='upper right', fontsize=9)
                ax_phase.grid(True, alpha=0.3)
            
            return ax,
        
        ani = FuncAnimation(fig, update, frames=len(self.t_history), 
                            interval=interval, blit=False)
        
        plt.tight_layout()
        return ani
    
    def create_interactive_plot(self):
        """Create interactive plot"""
        # Overall layout
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 1])
        
        # Main plot area
        ax_main = fig.add_subplot(gs[0, :])
        
        # Energy plot area
        ax_energy = fig.add_subplot(gs[1, 0])
        
        # Potential plot area
        ax_pot = fig.add_subplot(gs[1, 1])
        
        # Initial plot
        self.plot_results(frame=0, ax=ax_main, fig=fig)
        
        # Plot potential (オリジナルのスケールで表示)
        ax_pot.plot(self.x, self.V, 'r-')
        ax_pot.set_xlabel('Position x')
        ax_pot.set_ylabel('Potential V(x)')
        ax_pot.set_title('Potential (Original Scale)')
        ax_pot.set_ylim(0, np.max(self.V) * 1.1 if np.max(self.V) > 0 else 1.0)
        
        # 現在のエネルギーの表示
        psi_init = self.psi_history[0]
        energy = self.get_energy(psi_init)
        kinetic = self.hbar**2 * self.k0**2 / (2 * self.m)  # 近似的な運動エネルギー
        
        ax_energy.bar(['Total', 'Kinetic (Est.)'], [energy, kinetic], color=['blue', 'green'])
        ax_energy.axhline(y=self.V0, color='r', linestyle='--', label=f'Potential Height: {self.V0}')
        ax_energy.set_title('Energy Analysis')
        ax_energy.set_ylabel('Energy')
        ax_energy.legend()
        
        # スライダーエリアの設定
        slider_color = 'lightgoldenrodyellow'
        
        # Time slider
        ax_time = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=slider_color)
        time_slider = Slider(
            ax=ax_time,
            label='Time t',
            valmin=0,
            valmax=self.t_history[-1],
            valinit=0,
            valstep=self.dt*self.save_every
        )
        
        # Wave number (k0) slider
        ax_k0 = plt.axes([0.15, 0.1, 0.35, 0.03], facecolor=slider_color)
        k0_slider = Slider(
            ax=ax_k0,
            label='Wave Number k₀',
            valmin=1.0,
            valmax=10.0,
            valinit=self.k0
        )
        
        # Width of wave packet (sigma) slider
        ax_sigma = plt.axes([0.15, 0.05, 0.35, 0.03], facecolor=slider_color)
        sigma_slider = Slider(
            ax=ax_sigma,
            label='Wave Width σ',
            valmin=0.1,
            valmax=2.0,
            valinit=self.sigma
        )
        
        # Potential height (V0) slider
        ax_v0 = plt.axes([0.55, 0.1, 0.35, 0.03], facecolor=slider_color)
        v0_slider = Slider(
            ax=ax_v0,
            label='Potential Height V₀',
            valmin=0.1,
            valmax=10.0,
            valinit=self.V0
        )
        
        # Barrier width slider
        ax_width = plt.axes([0.55, 0.05, 0.35, 0.03], facecolor=slider_color)
        width_slider = Slider(
            ax=ax_width,
            label='Barrier Width',
            valmin=0.1,
            valmax=3.0,
            valinit=self.barrier_width
        )
        
        # リセットボタン
        ax_reset = plt.axes([0.85, 0.01, 0.1, 0.03])
        reset_button = Button(ax_reset, 'Reset', color=slider_color, hovercolor='0.975')
        
        # Time slider update function
        def update_time(val):
            idx = np.abs(self.t_history - time_slider.val).argmin()
            self.plot_results(frame=idx, ax=ax_main, fig=fig)
            fig.canvas.draw_idle()
        
        # K0, sigma, V0, widthスライダー更新関数
        def update_params(val=None):
            # 現在の設定を新しいシミュレーション用に保存
            config = {
                'x_min': self.x_min,
                'x_max': self.x_max,
                'nx': self.nx,
                'dt': self.dt,
                'nt': self.nt,
                'save_every': self.save_every,
                'x0': self.x0,
                'sigma': sigma_slider.val,
                'k0': k0_slider.val,
                'potential_type': self.potential_type,
                'V0': v0_slider.val,
                'barrier_width': width_slider.val,
                'barrier_pos': self.barrier_pos,
                'hbar': self.hbar,
                'm': self.m
            }
            
            # 新しいシミュレーションを作成して実行
            new_simulator = QuantumTunnelSimulator(config)
            new_simulator.time_evolve()
            
            # メインプロットを更新
            new_simulator.plot_results(frame=0, ax=ax_main, fig=fig)
            
            # ポテンシャルプロットを更新
            ax_pot.clear()
            ax_pot.plot(new_simulator.x, new_simulator.V, 'r-')
            ax_pot.set_xlabel('Position x')
            ax_pot.set_ylabel('Potential V(x)')
            ax_pot.set_title('Potential (Original Scale)')
            ax_pot.set_ylim(0, np.max(new_simulator.V) * 1.1 if np.max(new_simulator.V) > 0 else 1.0)
            
            # エネルギープロットを更新
            ax_energy.clear()
            psi_init = new_simulator.psi_history[0]
            energy = new_simulator.get_energy(psi_init)
            kinetic = new_simulator.hbar**2 * new_simulator.k0**2 / (2 * new_simulator.m)
            
            ax_energy.bar(['Total', 'Kinetic (Est.)'], [energy, kinetic], color=['blue', 'green'])
            ax_energy.axhline(y=new_simulator.V0, color='r', linestyle='--', label=f'Potential Height: {new_simulator.V0}')
            ax_energy.set_title('Energy Analysis')
            ax_energy.set_ylabel('Energy')
            ax_energy.legend()
            
            # タイムスライダーの更新
            time_slider.valmax = new_simulator.t_history[-1]
            time_slider.ax.set_xlim(0, new_simulator.t_history[-1])
            time_slider.set_val(0)
            
            # このシミュレーターのデータに更新
            self.__dict__.update(new_simulator.__dict__)
            
            fig.canvas.draw_idle()
        
        # リセット機能
        def reset(event):
            # 全てのスライダーを初期値に戻す
            k0_slider.reset()
            sigma_slider.reset()
            v0_slider.reset()
            width_slider.reset()
            time_slider.reset()
            update_params()
        
        # スライダーとボタンにイベントハンドラを接続
        time_slider.on_changed(update_time)
        k0_slider.on_changed(update_params)
        sigma_slider.on_changed(update_params)
        v0_slider.on_changed(update_params)
        width_slider.on_changed(update_params)
        reset_button.on_clicked(reset)
        
        # レイアウトの調整
        plt.tight_layout(rect=[0, 0.2, 1, 1])  # スライダー用のスペースを確保
        
        return fig, (time_slider, k0_slider, sigma_slider, v0_slider, width_slider)


def main():
    """Main execution function"""
    # Simulation settings
    config = {
        'x_min': -20.0,
        'x_max': 20.0,
        'nx': 1000,
        'dt': 0.01,
        'nt': 500,
        'save_every': 5,
        'x0': -7.0,
        'sigma': 1.0,
        'k0': 3.0,
        'potential_type': 'barrier',
        'V0': 5.0,
        'barrier_width': 1.0,
        'barrier_pos': 0.0,
        'hbar': 1.0,
        'm': 1.0
    }
    
    # Initialize simulator
    simulator = QuantumTunnelSimulator(config)
    
    # Calculate time evolution
    simulator.time_evolve()
    
    # Calculate transmission and reflection coefficients
    trans, refl = simulator.calculate_transmission_reflection()
    print(f"Transmission: {trans:.4f}")
    print(f"Reflection: {refl:.4f}")
    
    # Plot results (last frame)
    fig = simulator.plot_results(include_phase=True)
    plt.tight_layout()
    plt.savefig('quantum_tunnel_final.png', dpi=300)
    
    # Create and save animation
    ani = simulator.create_animation(interval=50, include_phase=True)
    # To save animation (requires ffmpeg)
    # ani.save('quantum_tunnel.mp4', writer='ffmpeg', dpi=300)
    
    # Show interactive plot
    fig_interactive, _ = simulator.create_interactive_plot()
    
    plt.show()


# Example of simulations with different potentials
def example_different_potentials():
    """Examples of simulations with different potentials"""
    # Barrier potential
    config_barrier = {
        'potential_type': 'barrier',
        'V0': 2.0,
        'barrier_width': 1.0,
        'k0': 2.0
    }
    
    # Well potential
    config_well = {
        'potential_type': 'well',
        'V0': 2.0,
        'barrier_width': 2.0,
        'k0': 2.0
    }
    
    # Double well potential
    config_double_well = {
        'potential_type': 'double_well',
        'V0': 2.0,
        'barrier_width': 4.0,
        'k0': 2.0
    }
    
    configs = [config_barrier, config_well, config_double_well]
    titles = ["Barrier Potential", "Well Potential", "Double Well Potential"]
    
    # Compare simulation results for each potential
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    for i, (config, title) in enumerate(zip(configs, titles)):
        simulator = QuantumTunnelSimulator(config)
        simulator.time_evolve()
        
        # Plot probability density and potential
        prob = simulator.get_probability_density()
        
        # 波動関数のプロット
        axes[i].plot(simulator.x, prob, 'b-', label='Probability Density |ψ|²')
        
        # ポテンシャル用の独立した軸を作成
        ax_potential = axes[i].twinx()
        ax_potential.plot(simulator.x, simulator.V, 'r-', label='Potential V(x)')
        ax_potential.set_ylabel('Potential V(x)', color='r')
        ax_potential.tick_params(axis='y', labelcolor='r')
        ax_potential.set_ylim(0, np.max(simulator.V) * 1.1 if np.max(simulator.V) > 0 else 1.0)
        
        # タイトルとラベルの設定
        axes[i].set_title(f"{title} - Transmission: {simulator.transmission:.4f}")
        axes[i].set_xlabel('Position x')
        axes[i].set_ylabel('Probability Density', color='b')
        axes[i].tick_params(axis='y', labelcolor='b')
        
        # 両方の軸のレジェンドを統合
        lines1, labels1 = axes[i].get_legend_handles_labels()
        lines2, labels2 = ax_potential.get_legend_handles_labels()
        axes[i].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('potential_comparison.png', dpi=300)
    plt.show()


# Example to investigate relationship between energy and transmission
def example_energy_vs_transmission():
    """Investigate relationship between energy (wave number k0) and transmission"""
    # Range of wave numbers
    k0_values = np.linspace(0.5, 5.0, 20)
    
    # Different barrier heights
    V0_values = [1.0, 2.0, 5.0]
    
    # Array to store results
    transmissions = np.zeros((len(V0_values), len(k0_values)))
    
    # Base settings
    base_config = {
        'x_min': -20.0,
        'x_max': 20.0,
        'nx': 1000,
        'dt': 0.01,
        'nt': 500,
        'save_every': 10,
        'x0': -7.0,
        'sigma': 1.0,
        'potential_type': 'barrier',
        'barrier_width': 1.0,
        'barrier_pos': 0.0
    }
    
    # プログレスの表示
    print("Starting energy vs transmission analysis...")
    total_runs = len(V0_values) * len(k0_values)
    run_count = 0
    
    # Run simulation for each combination of wave number and barrier height
    for i, V0 in enumerate(V0_values):
        for j, k0 in enumerate(k0_values):
            config = base_config.copy()
            config['k0'] = k0
            config['V0'] = V0
            
            # 進捗表示
            run_count += 1
            print(f"Running simulation {run_count}/{total_runs}: V0={V0}, k0={k0}")
            
            simulator = QuantumTunnelSimulator(config)
            simulator.time_evolve()
            
            transmissions[i, j] = simulator.transmission
    
    # Plot results with improved style
    plt.figure(figsize=(12, 8))
    
    # プロットスタイルの設定
    plt.style.use('seaborn-v0_8-darkgrid')
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # エネルギー計算とプロット
    hbar = simulator.hbar
    m = simulator.m
    
    for i, V0 in enumerate(V0_values):
        # Convert to energy (E = ħ²k²/2m)
        energies = hbar**2 * k0_values**2 / (2 * m)
        
        # V0をエネルギー単位で表示
        plt.axhline(y=0.5, color=colors[i], linestyle='--', alpha=0.3)
        plt.axvline(x=V0, color=colors[i], linestyle='--', alpha=0.3,
                    label=f'V₀ = {V0} (barrier height)')
        
        # 透過率プロット
        plt.plot(energies, transmissions[i], marker=markers[i], linestyle='-', 
                 color=colors[i], label=f'V₀ = {V0}', linewidth=2, markersize=8)
    
    # 理論曲線（参照用）
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Complete Transmission')
    
    # プロットの装飾
    plt.xlabel('Energy E = ħ²k²/2m', fontsize=12)
    plt.ylabel('Transmission Probability', fontsize=12)
    plt.title('Quantum Tunneling: Energy vs Transmission Probability', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(energies) * 1.05)
    plt.ylim(0, 1.05)
    
    # カラフルな背景で領域分け
    plt.axvspan(0, min(V0_values), alpha=0.1, color='red', label='Below all barriers')
    plt.axvspan(max(V0_values), max(energies), alpha=0.1, color='green', label='Above all barriers')
    
    plt.tight_layout()
    plt.savefig('energy_vs_transmission.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='量子トンネル効果シミュレーター')
    parser.add_argument('--mode', type=str, default='main',
                        choices=['main', 'potentials', 'energy', 'interactive'],
                        help='実行モード選択 (default: %(default)s)')
    parser.add_argument('--save', action='store_true',
                        help='アニメーションを保存する (ffmpegが必要)')
    parser.add_argument('--no-phase', action='store_true',
                        help='位相プロットを無効にする')
    
    args = parser.parse_args()
    
    # 選択したモードを実行
    if args.mode == 'main':
        print("基本的な量子トンネル効果シミュレーションを実行します...")
        main()
    elif args.mode == 'potentials':
        print("異なるポテンシャルでのシミュレーション比較を実行します...")
        example_different_potentials()
    elif args.mode == 'energy':
        print("エネルギーと透過率の関係を分析します...")
        example_energy_vs_transmission()
    elif args.mode == 'interactive':
        print("インタラクティブモードでシミュレーションを実行します...")
        # インタラクティブモード専用の設定
        config = {
            'x_min': -15.0,
            'x_max': 15.0,
            'nx': 1000,
            'dt': 0.01,
            'nt': 300,
            'save_every': 5,
            'x0': -5.0,
            'sigma': 0.7,
            'k0': 3.0,
            'potential_type': 'barrier',
            'V0': 3.0,
            'barrier_width': 0.8,
            'barrier_pos': 0.0,
            'hbar': 1.0,
            'm': 1.0
        }
        
        # シミュレーター初期化と実行
        simulator = QuantumTunnelSimulator(config)
        simulator.time_evolve()
        
        # ウェルカムメッセージ表示
        print("\n====== 量子トンネル効果インタラクティブシミュレーター ======")
        print("スライダーを動かして波動関数やポテンシャルの設定を変更できます。")
        print("- 波数(k₀): 粒子の運動量/エネルギーを調整")
        print("- 波束幅(σ): 位置の不確定性を調整")
        print("- ポテンシャルの高さ: バリアの高さを調整")
        print("- バリアの幅: バリアの厚さを調整")
        print("===============================================\n")
        
        # インタラクティブプロット表示
        fig_interactive, _ = simulator.create_interactive_plot()
        plt.show()
    
    # アニメーション保存オプション
    if args.save and args.mode == 'main':
        print("アニメーションをMP4形式で保存します...")
        config = {
            'x_min': -20.0,
            'x_max': 20.0,
            'nx': 1000,
            'dt': 0.01,
            'nt': 500,
            'save_every': 5,
            'x0': -7.0,
            'sigma': 1.0,
            'k0': 3.0,
            'potential_type': 'barrier',
            'V0': 5.0,
            'barrier_width': 1.0,
            'barrier_pos': 0.0,
        }
        
        simulator = QuantumTunnelSimulator(config)
        simulator.time_evolve()
        
        # アニメーション作成と保存
        include_phase = not args.no_phase
        ani = simulator.create_animation(interval=50, include_phase=include_phase)
        ani.save('quantum_tunnel.mp4', writer='ffmpeg', dpi=200)
        print("'quantum_tunnel.mp4'として保存しました。")
