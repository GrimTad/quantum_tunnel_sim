import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        
        # Plot potential (scaled)
        V_scaled = self.V / np.max(self.V) * np.max(prob) * 0.8 if np.max(self.V) > 0 else self.V
        ax.plot(self.x, V_scaled, 'r-', label='Potential V(x)')
        
        # Mark expected position
        expected_x = self.get_expected_position(psi)
        ax.axvline(x=expected_x, color='g', linestyle='--', label='Expected Position <x>')
        
        # Plot settings
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(0, max(np.max(prob) * 1.1, np.max(V_scaled) * 1.1))
        ax.set_xlabel('Position x')
        ax.set_ylabel('Probability Density / Potential (scaled)')
        ax.set_title(f'Time t = {self.t_history[frame]:.3f}')
        ax.legend(loc='upper right')
        
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
        """Create animation"""
        if include_phase:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax = ax1
            ax_phase = ax2
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax_phase = None
        
        plt.tight_layout()
        
        def update(frame):
            ax.clear()
            
            # Plot probability density
            psi = self.psi_history[frame]
            prob = self.get_probability_density(psi)
            ax.plot(self.x, prob, 'b-', label='Probability Density |ψ|²')
            
            # Plot potential (scaled)
            V_scaled = self.V / np.max(self.V) * np.max(prob) * 0.8 if np.max(self.V) > 0 else self.V
            ax.plot(self.x, V_scaled, 'r-', label='Potential V(x)')
            
            # Mark expected position
            expected_x = self.get_expected_position(psi)
            ax.axvline(x=expected_x, color='g', linestyle='--', label='Expected Position <x>')
            
            # Plot settings
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(0, max(np.max(self.get_probability_density(self.psi_history[0])) * 1.1, 
                              np.max(V_scaled) * 1.1))
            ax.set_xlabel('Position x')
            ax.set_ylabel('Probability Density / Potential (scaled)')
            ax.set_title(f'Time t = {self.t_history[frame]:.3f}')
            ax.legend(loc='upper right')
            
            # Plot phase (optional)
            if include_phase and ax_phase:
                ax_phase.clear()
                phase = self.get_phase(psi)
                ax_phase.plot(self.x, phase, 'g-', label='Phase arg(ψ)')
                ax_phase.set_xlabel('Position x')
                ax_phase.set_ylabel('Phase (rad)')
                ax_phase.set_ylim(-np.pi, np.pi)
                ax_phase.legend(loc='upper right')
            
            return ax,
        
        ani = FuncAnimation(fig, update, frames=len(self.t_history), 
                            interval=interval, blit=False)
        
        plt.tight_layout()
        return ani
    
    def create_interactive_plot(self):
        """Create interactive plot"""
        # Overall layout
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        
        # Main plot area
        ax_main = fig.add_subplot(gs[0, :])
        
        # Potential plot area
        ax_pot = fig.add_subplot(gs[1, 0])
        
        # Parameter control area
        ax_controls = fig.add_subplot(gs[1, 1])
        ax_controls.axis('off')
        
        # Initial plot
        self.plot_results(frame=0, ax=ax_main, fig=fig)
        
        # Plot potential
        ax_pot.plot(self.x, self.V, 'r-')
        ax_pot.set_xlabel('Position x')
        ax_pot.set_ylabel('Potential V(x)')
        ax_pot.set_title('Potential')
        
        # Add sliders
        plt.subplots_adjust(bottom=0.25)
        
        # Time slider
        ax_time = plt.axes([0.2, 0.15, 0.65, 0.03])
        time_slider = Slider(
            ax=ax_time,
            label='Time t',
            valmin=0,
            valmax=self.t_history[-1],
            valinit=0,
            valstep=self.dt*self.save_every
        )
        
        # Slider update function
        def update(val):
            # Find closest time index
            idx = np.abs(self.t_history - time_slider.val).argmin()
            self.plot_results(frame=idx, ax=ax_main, fig=fig)
            fig.canvas.draw_idle()
        
        time_slider.on_changed(update)
        
        # Fix figure layout
        plt.tight_layout(rect=[0, 0.2, 1, 1])  # Adjust to make room for slider
        return fig, time_slider


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
        
        # Scale potential
        V_scaled = simulator.V / np.max(simulator.V) * np.max(prob) * 0.8 if np.max(simulator.V) > 0 else simulator.V
        
        axes[i].plot(simulator.x, prob, 'b-', label='Probability Density |ψ|²')
        axes[i].plot(simulator.x, V_scaled, 'r-', label='Potential V(x)')
        axes[i].set_title(f"{title} - Transmission: {simulator.transmission:.4f}")
        axes[i].legend()
        axes[i].set_xlabel('Position x')
        axes[i].set_ylabel('Probability Density / Potential')
    
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
    
    # Run simulation for each combination of wave number and barrier height
    for i, V0 in enumerate(V0_values):
        for j, k0 in enumerate(k0_values):
            config = base_config.copy()
            config['k0'] = k0
            config['V0'] = V0
            
            simulator = QuantumTunnelSimulator(config)
            simulator.time_evolve()
            
            transmissions[i, j] = simulator.transmission
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    for i, V0 in enumerate(V0_values):
        # Convert to energy (E = ħ²k²/2m)
        energies = simulator.hbar**2 * k0_values**2 / (2 * simulator.m)
        plt.plot(energies, transmissions[i], 'o-', label=f'V0 = {V0}')
    
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Energy E = ħ²k²/2m')
    plt.ylabel('Transmission')
    plt.title('Relationship between Energy and Transmission')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_vs_transmission.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # To run other examples, uncomment these lines
    # example_different_potentials()
    # example_energy_vs_transmission()