"""Visualization module for creating plots and charts."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

logger = logging.getLogger(__name__)


class Visualizer:
    """Creates visualizations for the literature review results."""
    
    def __init__(self, config):
        """Initialize visualizer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.output_dir = Path(config.output_dir) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization settings
        self.format = config.viz_format
        self.dpi = config.viz_dpi
        self.style = config.viz_style
        self.figsize = config.viz_figsize
        self.colors = config.viz_colors
        
        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except:
            logger.warning(f"Style '{self.style}' not found, using default")
            plt.style.use('default')
    
    def create_all_visualizations(self, df: pd.DataFrame, save: bool = True) -> List[Path]:
        """Create all standard visualizations.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save figures to disk
            
        Returns:
            List of paths to saved figures
        """
        saved_figures = []
        
        # 1. Time series of publications
        fig_path = self.plot_time_series(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 2. Game type distribution
        fig_path = self.plot_game_types(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 3. AWScale histogram
        fig_path = self.plot_awscale_distribution(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 4. Failure modes chart
        fig_path = self.plot_failure_modes(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 5. LLM family distribution
        fig_path = self.plot_llm_families(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 6. Source database distribution
        fig_path = self.plot_source_distribution(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 7. Venue types
        fig_path = self.plot_venue_types(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        # 8. Open-ended vs Quantitative
        fig_path = self.plot_game_characteristics(df, save=save)
        if fig_path:
            saved_figures.append(fig_path)
        
        logger.info(f"Created {len(saved_figures)} visualizations")
        return saved_figures
    
    def plot_time_series(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot time series of publications.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Filter to valid years
            valid_years = df[df['year'] > 0]['year'].value_counts().sort_index()
            
            if len(valid_years) == 0:
                logger.warning("No valid years for time series plot")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot bars
            valid_years.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
            
            # Customize
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Number of Papers', fontsize=12)
            ax.set_title('Publications Over Time', fontsize=14, fontweight='bold')
            
            # Add trend line
            if len(valid_years) > 2:
                import numpy as np
                x = np.arange(len(valid_years))
                y = valid_years.values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, label=f'Trend')
                ax.legend()
            
            # Rotate x labels
            plt.xticks(rotation=45)
            
            # Tight layout
            plt.tight_layout()
            
            # Save
            if save:
                fig_path = self.output_dir / f'time_series.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved time series plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            plt.close()
            return None
    
    def plot_game_types(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot distribution of game types.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Get game type counts
            if 'game_type' not in df.columns:
                logger.warning("No game_type column for plot")
                return None
            
            game_counts = df['game_type'].value_counts()
            
            if len(game_counts) == 0:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Get colors for game types
            colors_map = self.colors.get('game_types', {})
            colors = [colors_map.get(game_type, 'gray') for game_type in game_counts.index]
            
            # Plot pie chart
            wedges, texts, autotexts = ax.pie(
                game_counts.values, 
                labels=game_counts.index,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Customize
            ax.set_title('Distribution of Game Types', fontsize=14, fontweight='bold')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            # Equal aspect ratio
            ax.axis('equal')
            
            # Save
            if save:
                fig_path = self.output_dir / f'game_types.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved game types plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating game types plot: {e}")
            plt.close()
            return None
    
    def plot_awscale_distribution(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot AWScale distribution.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Filter to papers with AWScale
            if 'awscale' not in df.columns:
                logger.warning("No awscale column for plot")
                return None
            
            awscale_data = df[df['awscale'].notna() & (df['awscale'] != '')]
            
            if len(awscale_data) == 0:
                return None
            
            # Convert to numeric
            awscale_values = pd.to_numeric(awscale_data['awscale'], errors='coerce').dropna()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Get colors
            colors_map = self.colors.get('awscale', {})
            bar_colors = [colors_map.get(str(i), 'gray') for i in range(1, 6)]
            
            # Create histogram
            counts = [sum(awscale_values == i) for i in range(1, 6)]
            bars = ax.bar(range(1, 6), counts, color=bar_colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom')
            
            # Customize
            ax.set_xlabel('AWScale Score', fontsize=12)
            ax.set_ylabel('Number of Papers', fontsize=12)
            ax.set_title('AWScale Distribution (Analytic ↔ Wild-Creative)', fontsize=14, fontweight='bold')
            ax.set_xticks(range(1, 6))
            ax.set_xticklabels([
                '1\n(Strictly\nAnalytic)', 
                '2\n(Mostly\nAnalytic)', 
                '3\n(Balanced)', 
                '4\n(Mostly\nCreative)', 
                '5\n(Wild-\nCreative)'
            ])
            
            # Add grid
            ax.yaxis.grid(True, alpha=0.3)
            
            # Set integer y-axis
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            # Tight layout
            plt.tight_layout()
            
            # Save
            if save:
                fig_path = self.output_dir / f'awscale_distribution.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved AWScale plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating AWScale plot: {e}")
            plt.close()
            return None
    
    def plot_failure_modes(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot failure modes distribution.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Get failure modes
            if 'failure_modes' not in df.columns:
                logger.warning("No failure_modes column for plot")
                return None
            
            # Count failure modes
            failure_counts = {}
            for modes_str in df['failure_modes'].dropna():
                if modes_str:
                    modes = modes_str.split('|')
                    for mode in modes:
                        mode = mode.strip()
                        if mode:
                            failure_counts[mode] = failure_counts.get(mode, 0) + 1
            
            if not failure_counts:
                return None
            
            # Sort by count
            sorted_modes = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 10
            top_modes = sorted_modes[:10]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 1.2))
            
            # Plot horizontal bars
            modes = [m[0] for m in top_modes]
            counts = [m[1] for m in top_modes]
            
            bars = ax.barh(modes, counts, color='coral', alpha=0.8)
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(count + 0.5, i, str(count), va='center')
            
            # Customize
            ax.set_xlabel('Number of Papers', fontsize=12)
            ax.set_title('Top Failure Modes Identified', fontsize=14, fontweight='bold')
            ax.invert_yaxis()  # Highest count at top
            
            # Add grid
            ax.xaxis.grid(True, alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save
            if save:
                fig_path = self.output_dir / f'failure_modes.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved failure modes plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating failure modes plot: {e}")
            plt.close()
            return None
    
    def plot_llm_families(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot LLM families distribution.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Get LLM family data
            if 'llm_family' not in df.columns:
                logger.warning("No llm_family column for plot")
                return None
            
            llm_counts = df['llm_family'].value_counts()
            
            if len(llm_counts) == 0:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot bars
            llm_counts[:10].plot(kind='bar', ax=ax, color='teal', alpha=0.8)
            
            # Customize
            ax.set_xlabel('LLM Family', fontsize=12)
            ax.set_ylabel('Number of Papers', fontsize=12)
            ax.set_title('Distribution of LLM Families Used', fontsize=14, fontweight='bold')
            
            # Rotate x labels
            plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.yaxis.grid(True, alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save
            if save:
                fig_path = self.output_dir / f'llm_families.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved LLM families plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating LLM families plot: {e}")
            plt.close()
            return None
    
    def plot_source_distribution(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot source database distribution.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Get source counts
            if 'source_db' not in df.columns:
                logger.warning("No source_db column for plot")
                return None
            
            source_counts = df['source_db'].value_counts()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot donut chart
            wedges, texts, autotexts = ax.pie(
                source_counts.values,
                labels=source_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.5)  # Makes it a donut
            )
            
            # Customize
            ax.set_title('Papers by Source Database', fontsize=14, fontweight='bold')
            
            # Save
            if save:
                fig_path = self.output_dir / f'source_distribution.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved source distribution plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating source distribution plot: {e}")
            plt.close()
            return None
    
    def plot_venue_types(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot venue types distribution.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Get venue type data
            if 'venue_type' not in df.columns:
                logger.warning("No venue_type column for plot")
                return None
            
            venue_counts = df['venue_type'].value_counts()
            
            if len(venue_counts) == 0:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot bars
            venue_counts.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
            
            # Customize
            ax.set_xlabel('Venue Type', fontsize=12)
            ax.set_ylabel('Number of Papers', fontsize=12)
            ax.set_title('Distribution of Venue Types', fontsize=14, fontweight='bold')
            
            # Rotate x labels if needed
            if len(venue_counts) > 4:
                plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.yaxis.grid(True, alpha=0.3)
            
            # Tight layout
            plt.tight_layout()
            
            # Save
            if save:
                fig_path = self.output_dir / f'venue_types.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved venue types plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating venue types plot: {e}")
            plt.close()
            return None
    
    def plot_game_characteristics(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot open-ended vs quantitative characteristics.
        
        Args:
            df: DataFrame with paper data
            save: Whether to save the figure
            
        Returns:
            Path to saved figure or None
        """
        try:
            # Check for required columns
            if 'open_ended' not in df.columns or 'quantitative' not in df.columns:
                logger.warning("Missing required columns for characteristics plot")
                return None
            
            # Count combinations
            char_data = df[['open_ended', 'quantitative']].value_counts()
            
            if len(char_data) == 0:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Prepare data for grouped bar chart
            categories = []
            open_ended_counts = []
            quantitative_counts = []
            both_counts = []
            neither_counts = []
            
            # Count each combination
            total = len(df)
            oe_yes_q_yes = len(df[(df['open_ended'] == 'yes') & (df['quantitative'] == 'yes')])
            oe_yes_q_no = len(df[(df['open_ended'] == 'yes') & (df['quantitative'] == 'no')])
            oe_no_q_yes = len(df[(df['open_ended'] == 'no') & (df['quantitative'] == 'yes')])
            oe_no_q_no = len(df[(df['open_ended'] == 'no') & (df['quantitative'] == 'no')])
            
            # Create stacked bar
            labels = ['Games']
            both = [oe_yes_q_yes]
            open_only = [oe_yes_q_no]
            quant_only = [oe_no_q_yes]
            neither = [oe_no_q_no]
            
            x = [0]
            width = 0.5
            
            # Plot stacked bars
            p1 = ax.bar(x, both, width, label='Both Open-ended & Quantitative', color='darkgreen')
            p2 = ax.bar(x, open_only, width, bottom=both, label='Open-ended Only', color='lightgreen')
            p3 = ax.bar(x, quant_only, width, bottom=[both[0] + open_only[0]], 
                       label='Quantitative Only', color='lightblue')
            p4 = ax.bar(x, neither, width, bottom=[both[0] + open_only[0] + quant_only[0]], 
                       label='Neither', color='lightgray')
            
            # Customize
            ax.set_ylabel('Number of Papers', fontsize=12)
            ax.set_title('Game Characteristics: Open-ended vs Quantitative', fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.legend(loc='upper right')
            
            # Add percentage labels
            total_height = both[0] + open_only[0] + quant_only[0] + neither[0]
            if total_height > 0:
                # Add text annotations
                if both[0] > 0:
                    ax.text(0, both[0]/2, f'{both[0]}\n({both[0]/total_height*100:.1f}%)', 
                           ha='center', va='center', fontweight='bold')
                if open_only[0] > 0:
                    ax.text(0, both[0] + open_only[0]/2, f'{open_only[0]}\n({open_only[0]/total_height*100:.1f}%)', 
                           ha='center', va='center', fontweight='bold')
                if quant_only[0] > 0:
                    ax.text(0, both[0] + open_only[0] + quant_only[0]/2, 
                           f'{quant_only[0]}\n({quant_only[0]/total_height*100:.1f}%)', 
                           ha='center', va='center', fontweight='bold')
                if neither[0] > 0:
                    ax.text(0, both[0] + open_only[0] + quant_only[0] + neither[0]/2, 
                           f'{neither[0]}\n({neither[0]/total_height*100:.1f}%)', 
                           ha='center', va='center', fontweight='bold')
            
            # Tight layout
            plt.tight_layout()
            
            # Save
            if save:
                fig_path = self.output_dir / f'game_characteristics.{self.format}'
                plt.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved game characteristics plot to {fig_path}")
                return fig_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Error creating game characteristics plot: {e}")
            plt.close()
            return None
    
    def create_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a summary report of the data.
        
        Args:
            df: DataFrame with paper data
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_papers': len(df),
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'year_range': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A',
            'sources': {}
        }
        
        # Source breakdown
        if 'source_db' in df.columns:
            summary['sources'] = df['source_db'].value_counts().to_dict()
        
        # Game type breakdown
        if 'game_type' in df.columns:
            summary['game_types'] = df['game_type'].value_counts().to_dict()
        
        # LLM breakdown
        if 'llm_family' in df.columns:
            summary['llm_families'] = df['llm_family'].value_counts().head(10).to_dict()
        
        # Failure modes
        if 'failure_modes' in df.columns:
            failure_counts = {}
            for modes_str in df['failure_modes'].dropna():
                if modes_str:
                    modes = modes_str.split('|')
                    for mode in modes:
                        mode = mode.strip()
                        if mode:
                            failure_counts[mode] = failure_counts.get(mode, 0) + 1
            summary['top_failure_modes'] = dict(sorted(failure_counts.items(), 
                                                      key=lambda x: x[1], 
                                                      reverse=True)[:10])
        
        # AWScale distribution
        if 'awscale' in df.columns:
            awscale_values = pd.to_numeric(df['awscale'], errors='coerce').dropna()
            if len(awscale_values) > 0:
                summary['awscale'] = {
                    'mean': float(awscale_values.mean()),
                    'median': float(awscale_values.median()),
                    'distribution': awscale_values.value_counts().sort_index().to_dict()
                }
        
        # Open-ended vs Quantitative
        if 'open_ended' in df.columns and 'quantitative' in df.columns:
            summary['game_characteristics'] = {
                'open_ended_yes': len(df[df['open_ended'] == 'yes']),
                'quantitative_yes': len(df[df['quantitative'] == 'yes']),
                'both': len(df[(df['open_ended'] == 'yes') & (df['quantitative'] == 'yes')]),
                'neither': len(df[(df['open_ended'] == 'no') & (df['quantitative'] == 'no')])
            }
        
        return summary