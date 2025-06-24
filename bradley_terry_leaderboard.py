"""
Bradley-Terry Model Leaderboard Analysis

METHODOLOGY OVERVIEW:
====================

This module implements a Bradley-Terry model to rank language models based on pairwise comparison data.
The Bradley-Terry model is a probabilistic approach for analyzing pairwise comparison outcomes.

MATHEMATICAL FOUNDATION:
-----------------------
The Bradley-Terry model assumes that for any two models i and j, the probability that model i beats model j is:
    P(i beats j) = π_i / (π_i + π_j)

Where π_i represents the "strength" parameter for model i. Taking the log-odds:
    log(P(i beats j) / P(j beats i)) = log(π_i) - log(π_j) = β_i - β_j

FEATURIZATION:
--------------
For each pairwise comparison between model A and model B:
1. Create a feature vector of length N (number of unique models)
2. Set position corresponding to model A = +1
3. Set position corresponding to model B = -1  
4. All other positions = 0
5. Target variable = 1 if model A wins, 0 if model B wins

Example: If we have models ["GPT-4", "Claude", "Gemini"] and compare GPT-4 vs Claude with GPT-4 winning:
- Feature vector: [1, -1, 0] (GPT-4=+1, Claude=-1, Gemini=0)
- Target: 1 (indicating GPT-4 wins)

OPERATIONS PERFORMED:
--------------------
1. DATA PREPROCESSING:
   - Load JSONL data containing pairwise comparisons
   - Filter to valid outcomes (A wins, B wins, Tie)
   - Remove ties for Bradley-Terry fitting (ties handled separately in statistics)

2. BRADLEY-TERRY MODEL FITTING:
   - Use logistic regression without intercept to fit: logit(P) = β_A - β_B
   - Coefficients β represent log-strength parameters for each model
   - Higher coefficient = stronger model on the evaluated axis

3. UNCERTAINTY QUANTIFICATION:
   - Bootstrap resampling (default: 100 iterations)
   - Calculate 95% confidence intervals for model strengths
   - Provides statistical significance testing

4. LEADERBOARD GENERATION:
   - Rank models by Bradley-Terry coefficients (higher = better)
   - Calculate win rates, battle counts, and tie rates
   - Generate visualizations and export results

5. BATTLE ANALYSIS:
   - Create pairwise battle count matrices
   - Calculate win rate matrices for head-to-head comparisons
   - Generate heatmap visualizations

OUTPUTS:
--------
- Ranked leaderboard with confidence intervals
- Battle count and win rate heatmaps
- CSV exports of all results
- Statistical summaries and visualizations

ASSUMPTIONS:
-----------
- Transitivity: If A > B and B > C, then A > C
- Independence: Each comparison is independent
- Consistency: Model strengths are stable across comparisons
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union
import argparse
import os
from pathlib import Path
import wandb

warnings.filterwarnings('ignore')


class BradleyTerryAnalyzer:
    """
    Bradley-Terry model analyzer for ranking language models from pairwise comparisons.
    
    This class provides a complete pipeline for:
    1. Loading and preprocessing pairwise comparison data
    2. Fitting Bradley-Terry models using logistic regression
    3. Performing bootstrap analysis for uncertainty quantification
    4. Generating leaderboards and visualizations
    """
    
    def __init__(self, data_path: str, regularization_strength: float = 1.0):
        """
        Initialize the Bradley-Terry analyzer.
        
        Args:
            data_path: Path to JSONL file containing pairwise model comparisons
            regularization_strength: L2 regularization strength (lower = more regularization)
                                   - 1.0 = standard regularization (default)
                                   - 0.1 = strong regularization (more stable, narrower CIs)
                                   - 0.01 = very strong regularization
                                   - 10.0 = weak regularization (less stable, wider CIs)
        """
        self.data_path = Path(data_path)
        self.regularization_strength = regularization_strength
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.models: Optional[List[str]] = None
        
        # Model components
        self.feature_matrix: Optional[np.ndarray] = None
        self.outcomes: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None
        
        # Results
        self.bootstrap_results: Optional[Dict] = None
        self.current_axis: Optional[str] = None
    
    def load_data(self, sample_size: Optional[int] = None) -> 'BradleyTerryAnalyzer':
        """
        Load pairwise comparison data from JSONL file.
        
        Args:
            sample_size: If provided, randomly sample this many rows for faster processing
            
        Returns:
            Self for method chaining
        """
        print(f"Loading data from {self.data_path}...")
        
        # Load JSONL data
        data_records = []
        with open(self.data_path, 'r') as file:
            for line in file:
                data_records.append(json.loads(line.strip()))
        
        self.raw_data = pd.DataFrame(data_records)
        
        # Apply sampling if requested
        if sample_size is not None and len(self.raw_data) > sample_size:
            self.raw_data = self.raw_data.sample(
                n=sample_size, random_state=42
            ).reset_index(drop=True)
            print(f"Sampled {sample_size} rows for faster processing")
        
        print(f"Loaded {len(self.raw_data)} total comparisons")
        print(f"Available columns: {list(self.raw_data.columns)}")
        
        return self
    
    def prepare_data(self, axis_column: str) -> 'BradleyTerryAnalyzer':
        """
        Prepare data for Bradley-Terry analysis on a specific evaluation axis.
        
        Args:
            axis_column: Name of the column containing comparison outcomes
            
        Returns:
            Self for method chaining
        """
        if self.raw_data is None:
            raise ValueError("Must call load_data() first")
            
        print(f"\nPreparing data for axis: {axis_column}")
        self.current_axis = axis_column
        
        # Validate column exists
        if axis_column not in self.raw_data.columns:
            available_cols = list(self.raw_data.columns)
            raise ValueError(f"Column '{axis_column}' not found. Available: {available_cols}")
        
        # Clean data: remove rows with missing required values
        required_columns = ['model_1_name', 'model_2_name', axis_column]
        clean_data = self.raw_data.dropna(subset=required_columns).copy()
        
        # Filter to valid outcomes
        valid_outcomes = ['A', 'B', 'Tie']
        clean_data = clean_data[clean_data[axis_column].isin(valid_outcomes)]
        
        if len(clean_data) == 0:
            raise ValueError(f"No valid comparisons found for axis '{axis_column}'")
        
        self.processed_data = clean_data
        
        # Extract unique models
        model_a_names = set(clean_data['model_1_name'].unique())
        model_b_names = set(clean_data['model_2_name'].unique())
        self.models = sorted(list(model_a_names | model_b_names))
        
        print(f"Valid comparisons: {len(clean_data)}")
        print(f"Outcome distribution: {clean_data[axis_column].value_counts().to_dict()}")
        print(f"Unique models: {len(self.models)}")
        
        return self
    
    def _build_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Bradley-Terry feature matrix and outcome vector.
        
        The feature matrix encodes each comparison as:
        - +1 for the position of model A
        - -1 for the position of model B  
        - 0 for all other model positions
        
        Returns:
            Tuple of (feature_matrix, outcomes) where outcomes is 1 if A wins, 0 if B wins
        """
        if self.processed_data is None or self.models is None or self.current_axis is None:
            raise ValueError("Must call prepare_data() first")
            
        print("Building Bradley-Terry feature representation...")
        
        # Remove ties for model fitting (standard Bradley-Terry approach)
        decisive_data = self.processed_data[
            self.processed_data[self.current_axis] != 'Tie'
        ].reset_index(drop=True)
        
        ties_excluded = len(self.processed_data) - len(decisive_data)
        print(f"Excluding {ties_excluded} ties, using {len(decisive_data)} decisive comparisons")
        
        # Initialize feature matrix and outcomes
        n_comparisons = len(decisive_data)
        n_models = len(self.models)
        feature_matrix = np.zeros((n_comparisons, n_models))
        outcomes = np.zeros(n_comparisons)
        
        # Create model name to index mapping
        model_to_index = {model: idx for idx, model in enumerate(self.models)}
        
        # Populate feature matrix
        for i, (_, row) in enumerate(decisive_data.iterrows()):
            model_a_idx = model_to_index[row['model_1_name']]
            model_b_idx = model_to_index[row['model_2_name']]
            
            # Bradley-Terry encoding: +1 for model A, -1 for model B
            feature_matrix[i, model_a_idx] = 1
            feature_matrix[i, model_b_idx] = -1
            
            # Outcome encoding: 1 if A wins, 0 if B wins
            outcomes[i] = 1 if row[self.current_axis] == 'A' else 0
        
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"A wins: {int(outcomes.sum())}, B wins: {int(len(outcomes) - outcomes.sum())}")
        
        return feature_matrix, outcomes
    
    def fit_model(self) -> 'BradleyTerryAnalyzer':
        """
        Fit the Bradley-Terry model using logistic regression.
        
        Returns:
            Self for method chaining
        """
        print("Fitting Bradley-Terry model...")
        
        # Build feature representation
        self.feature_matrix, self.outcomes = self._build_feature_matrix()
        
        # Fit logistic regression (no intercept needed for Bradley-Terry)
        model = LogisticRegression(
            fit_intercept=False,
            penalty='l2',
            C=self.regularization_strength,
            max_iter=1000,
            random_state=42
        )
        
        model.fit(self.feature_matrix, self.outcomes)
        self.coefficients = model.coef_[0]
        
        print("Model fitting completed")
        return self
    
    def bootstrap_uncertainty(self, n_iterations: int = 100) -> 'BradleyTerryAnalyzer':
        """
        Perform bootstrap analysis to quantify model ranking uncertainty.
        
        Args:
            n_iterations: Number of bootstrap samples to generate
            
        Returns:
            Self for method chaining
        """
        if self.feature_matrix is None or self.outcomes is None:
            raise ValueError("Must call fit_model() first")
            
        print(f"Running bootstrap analysis ({n_iterations} iterations)...")
        
        bootstrap_coefficients = []
        n_samples = len(self.outcomes)
        
        for i in range(n_iterations):
            if i % 25 == 0:
                print(f"  Progress: {i+1}/{n_iterations}")
            
            # Generate bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_X = self.feature_matrix[bootstrap_indices]
            bootstrap_y = self.outcomes[bootstrap_indices]
            
            # Fit model on bootstrap sample
            bootstrap_model = LogisticRegression(
                fit_intercept=False,
                penalty='l2', 
                C=self.regularization_strength,
                max_iter=1000,
                random_state=42
            )
            
            try:
                bootstrap_model.fit(bootstrap_X, bootstrap_y)
                bootstrap_coefficients.append(bootstrap_model.coef_[0])
            except Exception:
                # Skip failed fits (rare but can happen with extreme bootstrap samples)
                continue
        
        # Calculate bootstrap statistics
        bootstrap_coefficients = np.array(bootstrap_coefficients)
        self.bootstrap_results = {
            'coefficients': bootstrap_coefficients,
            'mean': np.mean(bootstrap_coefficients, axis=0),
            'std': np.std(bootstrap_coefficients, axis=0),
            'ci_lower': np.percentile(bootstrap_coefficients, 2.5, axis=0),
            'ci_upper': np.percentile(bootstrap_coefficients, 97.5, axis=0),
            'n_successful': len(bootstrap_coefficients)
        }
        
        print(f"Bootstrap completed: {self.bootstrap_results['n_successful']} successful iterations")
        return self
    
    def create_leaderboard(self) -> pd.DataFrame:
        """
        Generate the final leaderboard with rankings and statistics.
        
        Returns:
            DataFrame containing the complete leaderboard
        """
        if (self.models is None or self.coefficients is None or 
            self.bootstrap_results is None):
            raise ValueError("Must complete full analysis pipeline first")
            
        print("Generating leaderboard...")
        
        leaderboard_data = []
        
        for i, model in enumerate(self.models):
            # Calculate battle statistics
            stats = self._calculate_model_statistics(model)
            
            # Compile leaderboard entry
            entry = {
                'model': model,
                'coefficient': self.coefficients[i],
                'bootstrap_mean': self.bootstrap_results['mean'][i],
                'bootstrap_std': self.bootstrap_results['std'][i],
                'ci_lower': self.bootstrap_results['ci_lower'][i],
                'ci_upper': self.bootstrap_results['ci_upper'][i],
                **stats
            }
            leaderboard_data.append(entry)
        
        # Create and sort leaderboard
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('coefficient', ascending=False)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        return leaderboard
    
    def _calculate_model_statistics(self, model: str) -> Dict:
        """
        Calculate win/loss/tie statistics for a specific model.
        
        Args:
            model: Name of the model to analyze
            
        Returns:
            Dictionary containing battle statistics
        """
        if self.processed_data is None or self.current_axis is None:
            raise ValueError("Must call prepare_data() first")
            
        # Get all battles involving this model
        model_a_battles = self.processed_data['model_1_name'] == model
        model_b_battles = self.processed_data['model_2_name'] == model
        
        total_battles = model_a_battles.sum() + model_b_battles.sum()
        
        # Count outcomes when model is player A
        a_wins = (model_a_battles & (self.processed_data[self.current_axis] == 'A')).sum()
        a_ties = (model_a_battles & (self.processed_data[self.current_axis] == 'Tie')).sum()
        
        # Count outcomes when model is player B  
        b_wins = (model_b_battles & (self.processed_data[self.current_axis] == 'B')).sum()
        b_ties = (model_b_battles & (self.processed_data[self.current_axis] == 'Tie')).sum()
        
        total_wins = a_wins + b_wins
        total_ties = a_ties + b_ties
        
        return {
            'total_battles': total_battles,
            'total_wins': total_wins,
            'total_ties': total_ties,
            'win_rate': total_wins / total_battles if total_battles > 0 else 0,
            'tie_rate': total_ties / total_battles if total_battles > 0 else 0
        }
    
    def save_results(self, leaderboard: pd.DataFrame, output_dir: str) -> None:
        """
        Save all analysis results to files and log to wandb.
        
        Args:
            leaderboard: The leaderboard DataFrame to save
            output_dir: Directory to save results
        """
        if self.current_axis is None:
            raise ValueError("Must set current_axis first")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save leaderboard CSV
        csv_path = output_path / f'leaderboard_{self.current_axis}.csv'
        leaderboard.to_csv(csv_path, index=False)
        
        # Save leaderboard plot
        plot_path = output_path / f'leaderboard_{self.current_axis}.png'
        self._plot_leaderboard(leaderboard, str(plot_path))
        
        # Save win rate plot
        win_rate_plot_path = output_path / f'win_rate_{self.current_axis}.png'
        self._plot_win_rate(leaderboard, str(win_rate_plot_path))
        
        # Save non-ties plot
        non_ties_plot_path = output_path / f'non_ties_{self.current_axis}.png'
        self._plot_non_ties(leaderboard, str(non_ties_plot_path))
        
        # Save battle matrices
        battle_counts_path = output_path / f'battle_counts_{self.current_axis}.csv'
        win_rates_path = output_path / f'win_rates_{self.current_axis}.csv'
        self._save_battle_matrices(output_path)
        
        # Log to wandb if run is active
        if wandb.run is not None:
            print("Logging results to wandb...")
            
            # Log CSV files as artifacts
            wandb.log_artifact(str(csv_path), name=f'leaderboard_{self.current_axis}', type='dataset')
            wandb.log_artifact(str(battle_counts_path), name=f'battle_counts_{self.current_axis}', type='dataset')
            wandb.log_artifact(str(win_rates_path), name=f'win_rates_{self.current_axis}', type='dataset')
            
            # Log images to wandb
            wandb.log({
                f'{self.current_axis}_leaderboard': wandb.Image(str(plot_path), caption=f'Bradley-Terry Leaderboard: {self.current_axis.title()}'),
                f'{self.current_axis}_win_rates': wandb.Image(str(win_rate_plot_path), caption=f'Win Rates: {self.current_axis.title()}'),
                f'{self.current_axis}_non_ties': wandb.Image(str(non_ties_plot_path), caption=f'Decisive Battles: {self.current_axis.title()}')
            })
            
            # Log key metrics
            top_model = leaderboard.iloc[0]
            wandb.log({
                f'{self.current_axis}_top_model': top_model['model'],
                f'{self.current_axis}_top_coefficient': top_model['coefficient'],
                f'{self.current_axis}_top_win_rate': top_model['win_rate'],
                f'{self.current_axis}_num_models': len(leaderboard),
                f'{self.current_axis}_total_battles': leaderboard['total_battles'].sum() // 2,  # Divide by 2 to avoid double counting
                f'{self.current_axis}_regularization': self.regularization_strength
            })
        
        print(f"Results saved to: {output_path}")
    
    def _plot_leaderboard(self, leaderboard: pd.DataFrame, save_path: str) -> None:
        """Create and save leaderboard visualization."""
        if self.current_axis is None:
            raise ValueError("Must set current_axis first")
            
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        y_positions = np.arange(len(leaderboard))
        coefficients = leaderboard['coefficient'].values
        ci_lower = leaderboard['ci_lower'].values
        ci_upper = leaderboard['ci_upper'].values
        
        # Calculate error bar sizes
        lower_errors = coefficients - ci_lower
        upper_errors = ci_upper - coefficients
        
        # Create horizontal bar plot with confidence intervals
        plt.barh(y_positions, coefficients, 
                xerr=[lower_errors, upper_errors],
                capsize=5, alpha=0.7, color='skyblue')
        
        # Customize plot
        plt.yticks(y_positions, leaderboard['model'].values)
        plt.xlabel(f'Bradley-Terry Coefficient ({self.current_axis.title()})')
        plt.title(f'Model Leaderboard: {self.current_axis.title()}\n(with 95% Confidence Intervals)')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_win_rate(self, leaderboard: pd.DataFrame, save_path: str) -> None:
        """Create and save win rate visualization."""
        if self.current_axis is None:
            raise ValueError("Must set current_axis first")
            
        plt.figure(figsize=(12, 8))
        
        # Sort by win rate for this plot
        sorted_leaderboard = leaderboard.sort_values('win_rate', ascending=True)
        
        # Extract data for plotting
        y_positions = np.arange(len(sorted_leaderboard))
        win_rates = sorted_leaderboard['win_rate'].values * 100  # Convert to percentage
        
        # Create horizontal bar plot
        bars = plt.barh(y_positions, win_rates, alpha=0.7, color='lightgreen')
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, win_rates)):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{rate:.1f}%', ha='left', va='center', fontsize=9)
        
        # Customize plot
        plt.yticks(y_positions, sorted_leaderboard['model'].values)
        plt.xlabel('Win Rate (%)')
        plt.title(f'Model Win Rates: {self.current_axis.title()}')
        plt.grid(axis='x', alpha=0.3)
        plt.xlim(0, max(win_rates) * 1.1)  # Add some padding
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_non_ties(self, leaderboard: pd.DataFrame, save_path: str) -> None:
        """Create and save non-ties (decisive battles) visualization."""
        if self.current_axis is None:
            raise ValueError("Must set current_axis first")
            
        plt.figure(figsize=(12, 8))
        
        # Calculate non-ties (decisive battles)
        leaderboard_with_non_ties = leaderboard.copy()
        leaderboard_with_non_ties['non_ties'] = (
            leaderboard_with_non_ties['total_battles'] - leaderboard_with_non_ties['total_ties']
        )
        
        # Sort by non-ties for this plot
        sorted_leaderboard = leaderboard_with_non_ties.sort_values('non_ties', ascending=True)
        
        # Extract data for plotting
        y_positions = np.arange(len(sorted_leaderboard))
        non_ties = sorted_leaderboard['non_ties'].values
        
        # Create horizontal bar plot
        bars = plt.barh(y_positions, non_ties, alpha=0.7, color='lightcoral')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, non_ties)):
            plt.text(bar.get_width() + max(non_ties) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{int(count)}', ha='left', va='center', fontsize=9)
        
        # Customize plot
        plt.yticks(y_positions, sorted_leaderboard['model'].values)
        plt.xlabel('Number of Decisive Battles (Non-Ties)')
        plt.title(f'Model Decisive Battle Counts: {self.current_axis.title()}')
        plt.grid(axis='x', alpha=0.3)
        plt.xlim(0, max(non_ties) * 1.1)  # Add some padding
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_battle_matrices(self, output_dir: Path) -> None:
        """Generate and save battle count and win rate matrices."""
        if (self.processed_data is None or self.models is None or 
            self.current_axis is None):
            raise ValueError("Must complete data preparation first")
            
        print("Generating battle matrices...")
        
        # Create battle matrices
        n_models = len(self.models)
        battle_counts = np.zeros((n_models, n_models))
        win_counts = np.zeros((n_models, n_models))
        
        model_to_idx = {model: i for i, model in enumerate(self.models)}
        
        # Populate matrices
        for _, row in self.processed_data.iterrows():
            idx_a = model_to_idx[row['model_1_name']]
            idx_b = model_to_idx[row['model_2_name']]
            outcome = row[self.current_axis]
            
            # Count battles (symmetric)
            battle_counts[idx_a, idx_b] += 1
            battle_counts[idx_b, idx_a] += 1
            
            # Count wins (asymmetric)
            if outcome == 'A':
                win_counts[idx_a, idx_b] += 1
            elif outcome == 'B':
                win_counts[idx_b, idx_a] += 1
        
        # Create DataFrames
        battle_df = pd.DataFrame(battle_counts, index=self.models, columns=self.models)
        win_rate_df = pd.DataFrame(
            np.divide(win_counts, battle_counts, out=np.zeros_like(win_counts), where=battle_counts!=0),
            index=self.models, columns=self.models
        )
        
        # Save matrices
        battle_df.to_csv(output_dir / f'battle_counts_{self.current_axis}.csv')
        win_rate_df.to_csv(output_dir / f'win_rates_{self.current_axis}.csv')
    
    def print_summary(self, leaderboard: pd.DataFrame, top_n: int = 10) -> None:
        """Print a formatted summary of the leaderboard results."""
        if (self.current_axis is None or self.processed_data is None or 
            self.outcomes is None or self.models is None or 
            self.bootstrap_results is None):
            raise ValueError("Must complete full analysis pipeline first")
            
        print(f"\n{'='*80}")
        print(f"BRADLEY-TERRY LEADERBOARD: {self.current_axis.upper()}")
        print(f"{'='*80}")
        
        # Display top N models
        display_df = leaderboard.head(top_n).copy()
        
        # Format for display
        display_df['Coefficient'] = display_df['coefficient'].round(3)
        display_df['95% CI'] = display_df.apply(
            lambda row: f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]", axis=1
        )
        display_df['Win Rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
        display_df['Battles'] = display_df['total_battles']
        
        columns_to_show = ['rank', 'model', 'Coefficient', '95% CI', 'Win Rate', 'Battles']
        print(display_df[columns_to_show].to_string(index=False))
        
        # Summary statistics
        total_comparisons = len(self.processed_data)
        decisive_comparisons = len(self.outcomes)
        ties = total_comparisons - decisive_comparisons
        
        print(f"\nSummary:")
        print(f"  Total comparisons: {total_comparisons}")
        print(f"  Decisive comparisons: {decisive_comparisons}")
        print(f"  Ties: {ties}")
        print(f"  Models analyzed: {len(self.models)}")
        print(f"  Bootstrap iterations: {self.bootstrap_results['n_successful']}")
        print(f"  Regularization strength: {self.regularization_strength}")
        
        # Regularization guidance
        if self.regularization_strength >= 1.0:
            print(f"  Note: Standard/weak regularization - consider using 0.1 for subjective data")
        elif self.regularization_strength <= 0.01:
            print(f"  Note: Very strong regularization - rankings may be overly conservative")
        else:
            print(f"  Note: Good regularization for subjective evaluations")


def run_analysis(data_path: str, axis: str, output_dir: str = 'bt_results',
                n_bootstrap: int = 100, sample_size: Optional[int] = None,
                regularization_strength: float = 1.0, 
                wandb_project: Optional[str] = None) -> pd.DataFrame:
    """
    Run complete Bradley-Terry analysis for a single axis.
    
    Args:
        data_path: Path to JSONL data file
        axis: Name of the evaluation axis to analyze
        output_dir: Directory to save results
        n_bootstrap: Number of bootstrap iterations
        sample_size: Optional data sampling for faster processing
        regularization_strength: L2 regularization strength (lower = more regularization)
                               - 1.0 = standard (default)
                               - 0.1 = strong regularization (recommended for subjective data)
                               - 0.01 = very strong regularization
        wandb_project: Optional wandb project name for logging
        
    Returns:
        Final leaderboard DataFrame
    """
    # Initialize wandb run for this axis
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=axis,
            config={
                'axis': axis,
                'n_bootstrap': n_bootstrap,
                'sample_size': sample_size,
                'regularization_strength': regularization_strength,
                'data_path': str(data_path)
            },
            reinit=True  # Allow multiple runs in same script
        )
    
    try:
        # Initialize analyzer
        analyzer = BradleyTerryAnalyzer(data_path, regularization_strength=regularization_strength)
        
        # Run analysis pipeline
        leaderboard = (analyzer
                      .load_data(sample_size)
                      .prepare_data(axis)
                      .fit_model()
                      .bootstrap_uncertainty(n_bootstrap)
                      .create_leaderboard())
        
        # Display and save results
        analyzer.print_summary(leaderboard)
        
        axis_output_dir = os.path.join(output_dir, axis)
        analyzer.save_results(leaderboard, axis_output_dir)
        
        return leaderboard
    
    finally:
        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()


def main():
    """Command-line interface for Bradley-Terry analysis."""
    parser = argparse.ArgumentParser(
        description='Generate Bradley-Terry leaderboards for model comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single axis
  python bradley_terry_leaderboard.py --data comparisons.jsonl --axis helpfulness
  
  # Analyze with custom bootstrap iterations
  python bradley_terry_leaderboard.py --data comparisons.jsonl --axis empathy --bootstrap 200
  
  # Use data sampling for faster processing
  python bradley_terry_leaderboard.py --data comparisons.jsonl --axis creativity --sample 10000
  
  # Log to Weights & Biases (creates separate run for each axis)
  python bradley_terry_leaderboard.py --data comparisons.jsonl --wandb-project "model-evaluation"
  
  # Analyze single axis with wandb logging
  python bradley_terry_leaderboard.py --data comparisons.jsonl --axis humor --wandb-project "subjective-evals"
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to JSONL data file')
    parser.add_argument('--axis', help='Evaluation axis to analyze (if not provided, analyzes all common axes)')
    parser.add_argument('--bootstrap', type=int, default=100, help='Number of bootstrap iterations')
    parser.add_argument('--sample', type=int, help='Sample size for faster processing')
    parser.add_argument('--output', default='bt_results', help='Output directory for results')
    parser.add_argument('--regularization', type=float, default=0.5, 
                       help='L2 regularization strength (default: 0.5 for subjective data). '
                            'Lower values = more regularization = narrower confidence intervals. '
                            'Common values: 1.0 (standard), 0.5 (moderate), 0.1 (strong), 0.01 (very strong).')
    parser.add_argument('--wandb-project', type=str, default="vibes-bt-leaderboard",
                       help='Weights & Biases project name for logging results. If provided, creates a separate run for each axis.')
    
    args = parser.parse_args()
    
    # Define axes to analyze
    if args.axis:
        axes_to_analyze = [args.axis]
    else:
        # Default comprehensive set of evaluation axes
        axes_to_analyze = [
            'friendliness', 'formality', 'politeness', 'sycophancy', 'empathy', 
            'humor', 'anthropomorphism', 'assertiveness', 'directness', 'conciseness',
            'specificity', 'creativity', 'depth', 'relevance', 'context_awareness',
            'safety', 'refusal_to_answer', 'ethical_sensitivity', 'actionability',
            'user_intent_alignment', 'helpfulness', 'engagement', 'transparency', 'gen_z'
        ]
    
    # Run analysis for each axis
    print(f"Analyzing {len(axes_to_analyze)} evaluation axes...")
    print(f"Using regularization strength: {args.regularization}")
    if args.wandb_project:
        print(f"Logging to wandb project: {args.wandb_project}")
    
    for axis in axes_to_analyze:
        print(f"\n{'-'*60}")
        print(f"Processing axis: {axis}")
        print(f"{'-'*60}")
        
        try:
            run_analysis(
                data_path=args.data,
                axis=axis,
                output_dir=args.output,
                n_bootstrap=args.bootstrap,
                sample_size=args.sample,
                regularization_strength=args.regularization,
                wandb_project=args.wandb_project
            )
        except Exception as e:
            print(f"Error processing axis '{axis}': {e}")
            continue
    
    print(f"\n✓ Analysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main() 