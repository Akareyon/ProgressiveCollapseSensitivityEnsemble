import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import json

@dataclass
class CollapseParams:
    M0: float = 33e6
    m_ratio: float = 0.08
    h: float = 3.8
    E_abs: float = 1370.0
    alpha: float = 1.0
    lambda_: float = 0.18
    f_shed: float = 0.0
    CV_E: float = 0.0
    tau: float = 2.5
    N_lower: int = 97
    g: float = 9.81
    
    def __post_init__(self):
        self.h_eff = self.h * (1 - self.lambda_)


class WTCCollapseSimulator:
    def run_sim(self, p: CollapseParams, max_floors: int = 110, 
                rng: np.random.Generator = None) -> Tuple[bool, int, List[float]]:
        if rng is None:
            rng = np.random.default_rng()
        
        M = p.M0
        v = np.sqrt(2 * p.alpha * p.g * p.h_eff)
        
        floors_crushed = 0
        arrested = False
        energy_history = []
        
        for k in range(1, max_floors + 1):
            m = p.m_ratio * p.M0
            taper = 1 + (p.tau - 1) * (k / p.N_lower)
            E_abs_k = p.E_abs * taper * 1e6
            
            if p.CV_E > 0:
                log_std = np.sqrt(np.log(1 + p.CV_E**2))
                multiplier = np.exp(rng.normal(0, log_std))
                E_abs_k *= multiplier
            
            M_new = M + m * (1 - p.f_shed)
            u = (M / M_new) * v
            
            KE = 0.5 * M_new * u**2
            PE = M_new * p.g * p.h_eff
            E_avail = KE + PE
            energy_history.append(float(E_avail / 1e6))
            
            if E_avail <= E_abs_k:
                arrested = True
                floors_crushed = k
                break
            
            residual_KE = E_avail - E_abs_k
            v = np.sqrt(2 * residual_KE / M_new)
            M = M_new
            floors_crushed = k
        
        return arrested, floors_crushed, energy_history

    def make_param_distributions(self, n: int, rng: np.random.Generator = None) -> List[CollapseParams]:
        if rng is None:
            rng = np.random.default_rng()
        
        params = []
        for _ in range(n):
            p = CollapseParams(
                M0=rng.uniform(30e6, 58e6),
                m_ratio=rng.uniform(0.063, 0.10),
                E_abs=rng.uniform(250, 2000),
                alpha=rng.uniform(0.52, 1.0),
                lambda_=rng.uniform(0.15, 0.22),
                f_shed=rng.uniform(0.0, 0.50),
                CV_E=rng.uniform(0.0, 0.40)
            )
            params.append(p)
        return params

    def run_ensemble(self, n: int = 2000, seed: int = 42):
        rng = np.random.default_rng(seed)
        param_list = self.make_param_distributions(n, rng)
        
        results = []
        floors_list = []
        arrests = 0
        
        for p in param_list:
            arrested, floors, _ = self.run_sim(p, rng=rng)
            if arrested:
                arrests += 1
            floors_list.append(floors)
            
            results.append({
                'M0': float(p.M0)/1e6,
                'm_ratio': float(p.m_ratio),
                'E_abs': float(p.E_abs),
                'alpha': float(p.alpha),
                'lambda': float(p.lambda_),
                'f_shed': float(p.f_shed),
                'CV_E': float(p.CV_E),
                'arrested': arrested,
                'floors_crushed': floors
            })
        
        return {
            'results': results,
            'arrest_probability': arrests / n * 100,
            'floors_list': floors_list,
            'n': n
        }

    # ====================== PLOTTING FUNCTIONS ======================
    
    def plot_eabs_sweep(self, n_per_point=200, seed=42):
        """Arrest probability vs E_abs (one-at-a-time sweep)"""
        eabs_values = np.linspace(250, 2000, 20)
        arrest_probs = []
        
        rng = np.random.default_rng(seed)
        
        for eabs in eabs_values:
            count_arrest = 0
            for _ in range(n_per_point):
                p = CollapseParams(E_abs=eabs)
                arrested, _, _ = self.run_sim(p, rng=rng)
                if arrested:
                    count_arrest += 1
            arrest_probs.append(count_arrest / n_per_point * 100)
        
        plt.figure(figsize=(10, 6))
        plt.plot(eabs_values, arrest_probs, 'b-', linewidth=2, marker='o')
        plt.axvline(1370, color='red', linestyle='--', label='Schneider Threshold (1,370 MJ)')
        plt.axvspan(1000, 1300, alpha=0.1, color='orange', label='Bazant-Le range')
        plt.axvspan(1500, 2000, alpha=0.1, color='green', label='Korol-Sivakumaran range')
        plt.xlabel('E_abs (MJ)')
        plt.ylabel('Arrest Probability (%)')
        plt.title('Arrest Probability vs Energy Dissipation per Story')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_sensitivity(self, ensemble_results):
        """Bar chart of parameter correlations"""
        param_names = ['M0', 'm_ratio', 'E_abs', 'alpha', 'lambda', 'f_shed', 'CV_E']
        floors = ensemble_results['floors_list']
        results = ensemble_results['results']
        
        corrs = {}
        for name in param_names:
            vals = [r[name] for r in results]
            corr = np.corrcoef(vals, floors)[0, 1]
            corrs[name] = corr
        
        plt.figure(figsize=(10, 6))
        names = list(corrs.keys())
        values = [abs(corrs[k]) for k in names]
        colors = ['red' if k=='E_abs' else 'blue' for k in names]
        
        bars = plt.barh(names, values, color=colors)
        plt.xlabel('|Pearson Correlation| with Floors Crushed')
        plt.title('Parameter Sensitivity Analysis')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()

    def plot_floors_distribution(self, ensemble_results):
        """Histogram of floors crushed"""
        floors = np.array(ensemble_results['floors_list'])
        arrests = np.sum(np.array([r['arrested'] for r in ensemble_results['results']]))
        
        plt.figure(figsize=(10, 6))
        plt.hist(floors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(97, color='red', linestyle='--', label='Full Collapse')
        plt.xlabel('Floors Crushed')
        plt.ylabel('Number of Simulations')
        plt.title(f'Distribution of Floors Crushed (Arrest Rate: {ensemble_results["arrest_probability"]:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_eabs_m0_scatter(self, ensemble_results, n_samples=1000):
        """Scatter plot in (E_abs, M0) space colored by outcome"""
        results = ensemble_results['results'][:n_samples]
        eabs = [r['E_abs'] for r in results]
        m0 = [r['M0'] for r in results]
        arrested = [r['arrested'] for r in results]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(eabs, m0, c=arrested, cmap='RdYlGn_r', alpha=0.7, s=15)
        plt.colorbar(scatter, label='Arrested (1) / Collapse (0)')
        plt.axhline(33, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(58, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('E_abs (MJ)')
        plt.ylabel('M0 (×10⁶ kg)')
        plt.title('(E_abs, M0) Parameter Space')
        plt.grid(True, alpha=0.3)
        plt.show()


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    sim = WTCCollapseSimulator()
    
    print("Running Monte Carlo ensemble (N=2000)...")
    ensemble = sim.run_ensemble(n=2000, seed=42)
    
    print(f"Arrest Probability: {ensemble['arrest_probability']:.2f}%")
    
    # Generate all plots
    print("Generating plots...")
    sim.plot_eabs_sweep(n_per_point=300)
    sim.plot_sensitivity(ensemble)
    sim.plot_floors_distribution(ensemble)
    sim.plot_eabs_m0_scatter(ensemble)
    
    print("All plots generated!")
