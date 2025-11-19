#!/usr/bin/env python3
"""
Performance Model Validation for tiledCholesky
Compares theoretical vs actual slowdown across domain sizes

This script implements the saturation assumption validation described in
SATURATION_ASSUMPTION.md
"""

import json
import subprocess
import os
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class PerformanceValidator:
    def __init__(self, base_dir: str = ".", results_dir: str = "results/saturation_study", num_runs: int = 10):
        self.base_dir = Path(base_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Fixed parameters
        self.tile_size = 4
        self.num_runs = num_runs  # Number of runs for averaging runtime measurements

        # Domain sizes from small to large
        self.domain_sizes = [1024, 2048, 4096, 8192, 16384,
                             32768, 51200, 65536, 81920, 102400]

        # Results storage
        self.results = []

        # Paths
        # baseline uses tiledCholesky (respects optimize flag), optimized uses Ablation
        self.baseline_executable = self.base_dir / "build/userApplications/tiledCholesky"
        self.optimized_executable = self.base_dir / "build/userApplications/tiledCholeskyAblation"
        self.config_template = self.base_dir / "configs/config.json"

    def create_config(self, phase: str, n: int, memory_limit: Optional[float] = None) -> Dict:
        """
        Create configuration for different experiment phases

        Args:
            phase: "baseline", "phase1" (memory minimization), or "phase2" (performance opt)
            n: Matrix dimension
            memory_limit: Memory constraint in MiB (for phase2)
        """
        # Load default config as template
        default_config_path = self.base_dir / "defaultConfig.json"
        with open(default_config_path, 'r') as f:
            config = json.load(f)

        # Modify config based on phase
        # Fix device IDs to use device 0
        config['execution']['mainDeviceId'] = 0
        config['execution']['storageDeviceId'] = 1

        if phase == "baseline":
            config['generic']['optimize'] = False
            config['generic']['verify'] = True
            config['execution']['measurePeakMemoryUsage'] = True
            config['tiledCholesky']['n'] = n
            config['tiledCholesky']['t'] = self.tile_size
            return config
        elif phase == "phase1":
            config['generic']['optimize'] = True
            config['generic']['verify'] = True
            config['optimization']['firstStepSolverType'] = "BEAM_SEARCH"
            config['optimization']['secondStepSolverType'] = "MIP"
            config['optimization']['maxPeakMemoryUsageInMiB'] = 0  # No constraint
            config['optimization']['weightOfPeakMemoryUsage'] = 1.0
            config['optimization']['weightOfTotalRunningTime'] = 0.0
            config['optimization']['weightOfNumberOfMigrations'] = 0.0
            config['execution']['measurePeakMemoryUsage'] = True
            config['tiledCholesky']['n'] = n
            config['tiledCholesky']['t'] = self.tile_size
            return config
        elif phase == "phase2":
            assert memory_limit is not None, "phase2 requires memory_limit"
            config['generic']['optimize'] = True
            config['generic']['verify'] = True
            config['optimization']['firstStepSolverType'] = "BEAM_SEARCH"
            config['optimization']['secondStepSolverType'] = "MIP"
            config['optimization']['maxPeakMemoryUsageInMiB'] = memory_limit
            config['optimization']['weightOfPeakMemoryUsage'] = 0.0
            config['optimization']['weightOfTotalRunningTime'] = 1.0
            config['optimization']['weightOfNumberOfMigrations'] = 0.001
            config['execution']['measurePeakMemoryUsage'] = True
            config['tiledCholesky']['n'] = n
            config['tiledCholesky']['t'] = self.tile_size
            return config
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def run_experiment(self, executable: str, config_path: str, output_log: str) -> str:
        """Run executable with given config and capture output

        Args:
            executable: Path to executable (tiledCholeskyNaiveGraph or tiledCholeskyAblation)
            config_path: Path to config file
            output_log: Path to save output
        """
        # Read config to get N and T
        with open(config_path, 'r') as f:
            config = json.load(f)
        n = config['tiledCholesky']['n']
        t = config['tiledCholesky']['t']

        # Write config directly to config.json (don't use cp - ensures file is actually updated)
        target_config_path = self.base_dir / 'config.json'
        with open(config_path, 'r') as src:
            config_content = src.read()
        with open(target_config_path, 'w') as dst:
            dst.write(config_content)
            dst.flush()
            os.fsync(dst.fileno())  # Force write to disk

        # Verify the file was actually written correctly
        with open(target_config_path, 'r') as verify:
            written_content = verify.read()
        if written_content != config_content:
            raise RuntimeError(f"Config file verification failed! File not written correctly.")

        # Double-check the optimize flag is correct
        written_config = json.loads(written_content)
        expected_optimize = config['generic']['optimize']
        actual_optimize = written_config['generic']['optimize']
        if expected_optimize != actual_optimize:
            raise RuntimeError(f"Config optimize flag mismatch! Expected {expected_optimize}, got {actual_optimize}")

        print(f"  ✓ Config verified: optimize={actual_optimize}")

        # tiledCholeskyAblation uses --N= --T= format, tiledCholeskyNaiveGraph uses positional args
        if 'tiledCholeskyAblation' in executable:
            cmd = [
                "bash", "-c",
                f"source ~/miniconda3/bin/activate && conda activate frugal && "
                f"{executable} --N={n} --T={t}"
            ]
        else:
            # tiledCholeskyNaiveGraph uses positional arguments: executable N T
            cmd = [
                "bash", "-c",
                f"source ~/miniconda3/bin/activate && conda activate frugal && "
                f"{executable} {n} {t}"
            ]

        print(f"  Running: N={n}, T={t}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

        # Save output
        with open(output_log, 'w') as f:
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        return result.stdout

    def extract_metric(self, output: str, pattern: str, dtype=float) -> Optional[float]:
        """Extract a metric from output using regex"""
        match = re.search(pattern, output)
        if match:
            try:
                return dtype(match.group(1))
            except:
                return None
        return None

    def run_baseline(self, n: int) -> Dict:
        """Run baseline using tiledCholeskyNaiveGraph with multiple runs

        Runs dry run first (to match profiling overhead), then 10 measurement runs.
        """
        print(f"\n=== Baseline: N={n} (1 dry run + {self.num_runs} measurement runs) ===")

        # Create simple config
        config = self.create_config("baseline", n)

        # Safety check: verify optimize is False for baseline
        if config['generic']['optimize'] != False:
            raise ValueError(f"ERROR: Baseline config has optimize={config['generic']['optimize']}, expected False!")

        config_path = self.results_dir / f"config_baseline_{n}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  ✓ Config verified: optimize={config['generic']['optimize']}")

        # Dry run (discard - matches profiling overhead in optimized version)
        dry_run_log = self.results_dir / f"baseline_{n}_dryrun.log"
        print(f"  Dry run...")
        self.run_experiment(str(self.baseline_executable), str(config_path), str(dry_run_log))

        # Measurement runs
        runtimes = []
        managed_memory = None
        peak_memory = None

        for run_idx in range(self.num_runs):
            output_log = self.results_dir / f"baseline_{n}_run{run_idx}.log"
            output = self.run_experiment(str(self.baseline_executable), str(config_path), str(output_log))

            # Extract runtime - support both formats
            # Format 1: "Execution time: X.XXX ms" (tiledCholeskyNaiveGraph)
            # Format 2: "Total time used (s): X.XXXXX" (tiledCholesky)
            runtime = self.extract_metric(output, r'Execution time:\s+([\d.]+)\s+ms')
            if runtime is not None:
                runtimes.append(runtime / 1000.0)  # Convert to seconds
            else:
                runtime = self.extract_metric(output, r'Total time used \(s\):\s+([\d.]+)')
                if runtime is not None:
                    runtimes.append(runtime)  # Already in seconds

            # Memory metrics from first run
            if managed_memory is None:
                # Managed memory (matrix tiles only) - support both formats
                # Format 1: "Total managed memory: X.XX MB" (tiledCholeskyNaiveGraph)
                # Format 2: "[MEMORY-INFO] Total managed memory size: X.XX MB" (tiledCholesky)
                managed_memory = self.extract_metric(output, r'Total managed memory:\s+([\d.]+)\s+MB')
                if managed_memory is None:
                    managed_memory = self.extract_metric(output, r'\[MEMORY-INFO\] Total managed memory size:\s+([\d.]+)\s+MB')

            if peak_memory is None:
                # Peak GPU memory (total including workspace, libraries, etc.) - support both formats
                # Format 1: "Peak GPU memory usage during execution: X.XX MB" (tiledCholeskyNaiveGraph)
                # Format 2: "Peak memory usage (MiB): X.XX" (tiledCholesky)
                peak_memory = self.extract_metric(output, r'Peak GPU memory usage during execution:\s+([\d.]+)\s+MB')
                if peak_memory is None:
                    peak_memory = self.extract_metric(output, r'Peak memory usage \(MiB\):\s+([\d.]+)')

        # Calculate statistics
        runtime_avg = np.mean(runtimes) if runtimes else None
        runtime_std = np.std(runtimes) if runtimes else None

        tflops = None
        if runtime_avg:
            tflops = (n**3 / 3) / (runtime_avg * 1e12)

        if runtime_avg is not None and peak_memory is not None:
            print(f"  Runtime: {runtime_avg:.3f}s ± {runtime_std:.3f}s ({len(runtimes)}/{self.num_runs} runs)")
            print(f"  Managed Memory: {managed_memory:.2f}MB, Peak Memory: {peak_memory:.2f}MB, TFLOPS: {tflops:.2f}")
        else:
            print(f"  Runtime: {runtime_avg}, Peak Memory: {peak_memory}, TFLOPS: {tflops} (Failed to extract metrics)")

        return {
            'runtime': runtime_avg,
            'runtime_std': runtime_std,
            'managed_memory': managed_memory,
            'peak_memory': peak_memory,
            'tflops': tflops
        }

    def run_phase1_minimize_memory(self, n: int, baseline: Dict) -> Dict:
        """Phase 1: Find minimal memory"""
        print(f"\n=== Phase 1 (Memory Minimization): N={n} ===")

        # Create config
        config = self.create_config("phase1", n)

        # Safety check: verify optimize is True for phase1
        if config['generic']['optimize'] != True:
            raise ValueError(f"ERROR: Phase1 config has optimize={config['generic']['optimize']}, expected True!")

        config_path = self.results_dir / f"config_phase1_{n}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  ✓ Config verified: optimize={config['generic']['optimize']}")

        # Run
        output_log = self.results_dir / f"phase1_{n}.log"
        output = self.run_experiment(str(self.optimized_executable), str(config_path), str(output_log))

        # Extract managed memory (optimized)
        optimized_managed = self.extract_metric(output, r'Optimized peak memory usage \(MiB\):\s+([\d.]+)')

        # Extract actual peak memory during execution
        actual_peak = self.extract_metric(output, r'Peak GPU memory usage during execution:\s+([\d.]+)\s+MB')

        migrations = self.extract_metric(output, r'Number of migrations:\s+(\d+)', int)

        # Calculate BOTH memory reductions
        managed_reduction = 0.0
        peak_reduction = 0.0

        if optimized_managed and baseline.get('managed_memory'):
            managed_reduction = (1 - optimized_managed / baseline['managed_memory']) * 100

        if actual_peak and baseline.get('peak_memory'):
            peak_reduction = (1 - actual_peak / baseline['peak_memory']) * 100

        # Print results
        managed_mem_str = f"{baseline['managed_memory']:.2f}" if baseline.get('managed_memory') is not None else "N/A"
        peak_mem_str = f"{baseline['peak_memory']:.2f}" if baseline.get('peak_memory') is not None else "N/A"
        print(f"  Baseline - Managed: {managed_mem_str}MB, Peak: {peak_mem_str}MB")
        if optimized_managed is not None:
            print(f"  Optimized - Managed: {optimized_managed:.2f}MB (reduction: {managed_reduction:.1f}%), Peak: {actual_peak:.2f}MB (reduction: {peak_reduction:.1f}%)")
            print(f"  Migrations: {migrations if migrations else 0}")
        else:
            print(f"  Failed to extract optimized memory metrics")

        return {
            'optimized_managed': optimized_managed,
            'actual_peak': actual_peak,
            'managed_reduction': managed_reduction,
            'peak_reduction': peak_reduction,
            'migrations': migrations if migrations else 0
        }

    def run_phase2_optimize_performance(self, n: int, memory_limit: float, baseline: Dict) -> Dict:
        """Phase 2: Optimize performance within memory constraint with multiple runs for averaging"""
        if memory_limit:
            print(f"\n=== Phase 2 (Performance Optimization): N={n}, Memory Limit={memory_limit:.2f}MiB (averaging over {self.num_runs} runs) ===")
        else:
            print(f"\n=== Phase 2 (Performance Optimization): N={n}, Memory Limit={memory_limit} (SKIP) ===")
            return {
                'predicted_runtime': None,
                'actual_runtime': None,
                'actual_runtime_std': None,
                'predicted_slowdown': None,
                'actual_slowdown': None,
                'prediction_error': None,
                'final_memory': None,
                'memory_reduction_maintained': None,
                'optimized_tflops': None
            }

        # Create config
        config = self.create_config("phase2", n, memory_limit)

        # Safety check: verify optimize is True for phase2
        if config['generic']['optimize'] != True:
            raise ValueError(f"ERROR: Phase2 config has optimize={config['generic']['optimize']}, expected True!")

        config_path = self.results_dir / f"config_phase2_{n}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  ✓ Config verified: optimize={config['generic']['optimize']}")

        # Run multiple times and collect runtime measurements
        pred_runtime = None
        optimized_managed = None
        actual_peak = None
        actual_runtimes = []

        for run_idx in range(self.num_runs):
            output_log = self.results_dir / f"phase2_{n}_run{run_idx}.log"
            output = self.run_experiment(str(self.optimized_executable), str(config_path), str(output_log))

            # Extract predicted runtime (deterministic, from first run)
            if pred_runtime is None:
                pred_runtime = self.extract_metric(output, r'Predicted runtime:\s+([\d.]+)\s+s')
                if pred_runtime is None:
                    pred_runtime = self.extract_metric(output, r'Longest path in optimized graph \(s\):\s+([\d.]+)')
                if pred_runtime is None:
                    pred_runtime = self.extract_metric(output, r'Total running time \(s\):\s+([\d.]+)')

            # Extract actual runtime from GPU execution
            actual_runtime = self.extract_metric(output, r'Execution time:\s+([\d.]+)\s+ms')
            if actual_runtime is not None:
                actual_runtimes.append(actual_runtime / 1000.0)

            # Memory metrics - deterministic (from first run)
            if optimized_managed is None:
                optimized_managed = self.extract_metric(output, r'Optimized peak memory usage \(MiB\):\s+([\d.]+)')
            if actual_peak is None:
                actual_peak = self.extract_metric(output, r'Peak GPU memory usage during execution:\s+([\d.]+)\s+MB')

        # Calculate average actual runtime
        actual_runtime_avg = np.mean(actual_runtimes) if actual_runtimes else None
        actual_runtime_std = np.std(actual_runtimes) if actual_runtimes else None

        # Calculate derived metrics
        pred_slowdown = None
        actual_slowdown = None
        prediction_error = None
        managed_reduction = 0.0
        peak_reduction = 0.0
        actual_tflops = None

        if pred_runtime and baseline['runtime']:
            pred_slowdown = pred_runtime / baseline['runtime']

        if actual_runtime_avg and baseline['runtime']:
            actual_slowdown = actual_runtime_avg / baseline['runtime']
            actual_tflops = (n**3 / 3) / (actual_runtime_avg * 1e12)

        if pred_runtime and actual_runtime_avg:
            prediction_error = abs(pred_runtime - actual_runtime_avg) / actual_runtime_avg * 100

        # Calculate BOTH memory reductions
        if optimized_managed and baseline.get('managed_memory'):
            managed_reduction = (1 - optimized_managed / baseline['managed_memory']) * 100

        if actual_peak and baseline.get('peak_memory'):
            peak_reduction = (1 - actual_peak / baseline['peak_memory']) * 100

        # Print results with proper None handling
        if pred_runtime is not None and pred_slowdown is not None:
            print(f"  Predicted: {pred_runtime:.3f}s ({pred_slowdown:.2f}x slowdown)")
        else:
            print(f"  Predicted: N/A")

        if actual_runtime_avg is not None and actual_slowdown is not None and actual_runtime_std is not None:
            print(f"  Actual: {actual_runtime_avg:.3f}s ± {actual_runtime_std:.3f}s ({actual_slowdown:.2f}x slowdown) [{len(actual_runtimes)}/{self.num_runs} runs]")
        else:
            print(f"  Actual: N/A (Failed to extract metrics)")

        if prediction_error is not None:
            print(f"  Prediction Error: {prediction_error:.1f}%")
        else:
            print(f"  Prediction Error: N/A")

        # Print both memory reductions
        print(f"  Managed Memory Reduction: {managed_reduction:.1f}%")
        print(f"  Peak Memory Reduction: {peak_reduction:.1f}%")

        if actual_tflops is not None:
            print(f"  TFLOPS: {actual_tflops:.2f}")
        else:
            print(f"  TFLOPS: N/A")

        return {
            'predicted_runtime': pred_runtime,
            'actual_runtime': actual_runtime_avg,
            'actual_runtime_std': actual_runtime_std,
            'predicted_slowdown': pred_slowdown,
            'actual_slowdown': actual_slowdown,
            'prediction_error': prediction_error,
            'optimized_managed': optimized_managed,
            'actual_peak': actual_peak,
            'managed_reduction': managed_reduction,
            'peak_reduction': peak_reduction,
            'optimized_tflops': actual_tflops
        }

    def run_full_experiment(self, n: int):
        """Run all phases for a given domain size"""
        print(f"\n{'='*60}")
        print(f"RUNNING FULL EXPERIMENT: N={n}, T={self.tile_size}")
        print(f"{'='*60}")

        try:
            # Baseline
            baseline = self.run_baseline(n)

            # Phase 1: Find minimal memory
            phase1 = self.run_phase1_minimize_memory(n, baseline)

            # Phase 2: Optimize performance at minimal memory
            phase2 = self.run_phase2_optimize_performance(n, phase1['optimized_managed'], baseline)

            # Store all results
            result = {
                'n': n,
                't': self.tile_size,
                'baseline_runtime': baseline['runtime'],
                'baseline_managed_memory': baseline['managed_memory'],
                'baseline_peak_memory': baseline['peak_memory'],
                'baseline_tflops': baseline['tflops'],
                **phase1,
                **phase2
            }

            self.results.append(result)

            # Save intermediate results
            self.save_results()

            print(f"\n✅ Completed N={n}")

        except Exception as e:
            print(f"\n❌ Error for N={n}: {e}")
            import traceback
            traceback.print_exc()

    def save_results(self):
        """Save results to CSV and JSON"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        # Save CSV
        csv_path = self.results_dir / "performance_validation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to: {csv_path}")

        # Save JSON
        json_path = self.results_dir / "performance_validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Print summary table with better formatting
        print("\n=== SUMMARY TABLE ===")
        print(f"{'N':<8} {'Managed Red (%)':<15} {'Peak Red (%)':<15} {'Pred Slow (x)':<15} {'Actual Slow (x)':<16} {'Pred Err (%)':<13} {'Base TFLOPS':<13} {'Opt TFLOPS':<13}")
        print("=" * 120)
        for _, row in df.iterrows():
            # Format with proper None handling
            managed_red = f"{row['managed_reduction']:.1f}" if pd.notna(row.get('managed_reduction')) else "N/A"
            peak_red = f"{row['peak_reduction']:.1f}" if pd.notna(row.get('peak_reduction')) else "N/A"
            pred_slow = f"{row['predicted_slowdown']:.2f}" if pd.notna(row.get('predicted_slowdown')) else "N/A"
            actual_slow = f"{row['actual_slowdown']:.2f}" if pd.notna(row.get('actual_slowdown')) else "N/A"
            pred_err = f"{row['prediction_error']:.1f}" if pd.notna(row.get('prediction_error')) else "N/A"
            base_tflops = f"{row['baseline_tflops']:.2f}" if pd.notna(row.get('baseline_tflops')) else "N/A"
            opt_tflops = f"{row['optimized_tflops']:.2f}" if pd.notna(row.get('optimized_tflops')) else "N/A"

            print(f"{row['n']:<8.0f} {managed_red:<15} {peak_red:<15} {pred_slow:<15} {actual_slow:<16} {pred_err:<13} {base_tflops:<13} {opt_tflops:<13}")

        print("\nNote: Managed Red = managed memory reduction, Peak Red = total GPU memory reduction")
        print("Slowdown values are multiplicative factors (e.g., 1.5x means 1.5 times slower)")

    def run_baseline_only(self, domain_sizes: Optional[List[int]] = None):
        """Run ONLY baseline experiments (separate invocation with optimize=false)"""
        sizes = domain_sizes if domain_sizes else self.domain_sizes

        print(f"\n🚀 Running BASELINE-ONLY Mode")
        print(f"Domain sizes: {sizes}")
        print(f"Tile size: {self.tile_size}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Results directory: {self.results_dir}")

        # Set optimize=false in config.json for entire run
        baseline_config = self.create_config("baseline", sizes[0])
        baseline_config_path = self.results_dir / "config_baseline_global.json"
        with open(baseline_config_path, 'w') as f:
            json.dump(baseline_config, f, indent=2)
        import subprocess
        subprocess.run(['cp', str(baseline_config_path), 'config.json'], check=True, cwd=self.base_dir)
        subprocess.run(['sync'], check=False)
        print(f"✓ Global config.json set to optimize=false\n")

        for n in sizes:
            print(f"\n{'='*60}")
            print(f"BASELINE: N={n}, T={self.tile_size}")
            print(f"{'='*60}")
            try:
                baseline = self.run_baseline(n)
                # Save baseline results to JSON for later merging
                baseline_file = self.results_dir / f"baseline_{n}.json"
                with open(baseline_file, 'w') as f:
                    json.dump(baseline, f, indent=2)
                print(f"✅ Baseline for N={n} saved")
            except Exception as e:
                print(f"❌ Baseline failed for N={n}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ All baselines completed!")
        print(f"📊 Results saved in: {self.results_dir}")

    def run_optimized_only(self, domain_sizes: Optional[List[int]] = None):
        """Run ONLY optimized experiments (separate invocation with optimize=true)"""
        sizes = domain_sizes if domain_sizes else self.domain_sizes

        print(f"\n🚀 Running OPTIMIZED-ONLY Mode")
        print(f"Domain sizes: {sizes}")
        print(f"Tile size: {self.tile_size}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Results directory: {self.results_dir}")

        # Set optimize=true in config.json for entire run
        optimized_config = self.create_config("phase1", sizes[0])
        optimized_config_path = self.results_dir / "config_optimized_global.json"
        with open(optimized_config_path, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        import subprocess
        subprocess.run(['cp', str(optimized_config_path), 'config.json'], check=True, cwd=self.base_dir)
        subprocess.run(['sync'], check=False)
        print(f"✓ Global config.json set to optimize=true\n")

        for n in sizes:
            # Load baseline results
            baseline_file = self.results_dir / f"baseline_{n}.json"
            if not baseline_file.exists():
                print(f"⚠️  Warning: No baseline found for N={n}, skipping optimized experiments")
                continue

            with open(baseline_file, 'r') as f:
                baseline = json.load(f)

            print(f"\n{'='*60}")
            print(f"OPTIMIZED EXPERIMENT: N={n}, T={self.tile_size}")
            print(f"{'='*60}")

            try:
                # Phase 1: Find minimal memory
                phase1 = self.run_phase1_minimize_memory(n, baseline)

                # Phase 2: Optimize performance at minimal memory
                phase2 = self.run_phase2_optimize_performance(n, phase1['optimized_managed'], baseline)

                # Store all results
                result = {
                    'n': n,
                    't': self.tile_size,
                    'baseline_runtime': baseline['runtime'],
                    'baseline_managed_memory': baseline['managed_memory'],
                    'baseline_peak_memory': baseline['peak_memory'],
                    'baseline_tflops': baseline['tflops'],
                    **phase1,
                    **phase2
                }

                self.results.append(result)
                self.save_results()

                print(f"\n✅ Completed N={n}")

            except Exception as e:
                print(f"\n❌ Error for N={n}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ All optimized experiments completed!")
        print(f"📊 Results saved in: {self.results_dir}")

    def run_all(self, domain_sizes: Optional[List[int]] = None):
        """Run validation for all domain sizes

        Modified to run all baselines first, then all optimized experiments.
        This ensures config.json filesystem sync is not an issue.
        """
        sizes = domain_sizes if domain_sizes else self.domain_sizes

        print(f"\n🚀 Starting Performance Validation")
        print(f"Domain sizes: {sizes}")
        print(f"Tile size: {self.tile_size}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Results directory: {self.results_dir}")

        # Store baseline results for each size
        baseline_results = {}

        # STEP 1: Run ALL baselines first (with optimize=false)
        print(f"\n{'='*60}")
        print("STEP 1: Running ALL BASELINE experiments (optimize=false)")
        print(f"{'='*60}")

        # CRITICAL: Set optimize=false in config.json BEFORE running baselines
        # This ensures all baselines use the same config state
        baseline_config = self.create_config("baseline", sizes[0])
        baseline_config_path = self.results_dir / "config_baseline_initial.json"
        with open(baseline_config_path, 'w') as f:
            json.dump(baseline_config, f, indent=2)
        # Copy to main config location
        import subprocess
        subprocess.run(['cp', str(baseline_config_path), 'config.json'], check=True, cwd=self.base_dir)
        subprocess.run(['sync'], check=False)
        print(f"✓ Set config.json to optimize=false before baseline batch")

        for n in sizes:
            print(f"\n--- Baseline for N={n} ---")
            try:
                baseline_results[n] = self.run_baseline(n)
            except Exception as e:
                print(f"❌ Baseline failed for N={n}: {e}")
                import traceback
                traceback.print_exc()
                baseline_results[n] = None

        # STEP 2: Run ALL optimized experiments (with optimize=true)
        print(f"\n{'='*60}")
        print("STEP 2: Running ALL OPTIMIZED experiments (optimize=true)")
        print(f"{'='*60}")

        # CRITICAL: Set optimize=true in config.json BEFORE running optimized experiments
        optimized_config = self.create_config("phase1", sizes[0])
        optimized_config_path = self.results_dir / "config_optimized_initial.json"
        with open(optimized_config_path, 'w') as f:
            json.dump(optimized_config, f, indent=2)
        # Copy to main config location
        subprocess.run(['cp', str(optimized_config_path), 'config.json'], check=True, cwd=self.base_dir)
        subprocess.run(['sync'], check=False)
        print(f"✓ Set config.json to optimize=true before optimized batch")

        for n in sizes:
            print(f"\n{'='*60}")
            print(f"OPTIMIZED EXPERIMENT: N={n}, T={self.tile_size}")
            print(f"{'='*60}")

            baseline = baseline_results.get(n)
            if baseline is None:
                print(f"⚠️  Skipping N={n} - baseline failed")
                continue

            try:
                # Phase 1: Find minimal memory
                phase1 = self.run_phase1_minimize_memory(n, baseline)

                # Phase 2: Optimize performance at minimal memory
                phase2 = self.run_phase2_optimize_performance(n, phase1['optimized_managed'], baseline)

                # Store all results
                result = {
                    'n': n,
                    't': self.tile_size,
                    'baseline_runtime': baseline['runtime'],
                    'baseline_managed_memory': baseline['managed_memory'],
                    'baseline_peak_memory': baseline['peak_memory'],
                    'baseline_tflops': baseline['tflops'],
                    **phase1,
                    **phase2
                }

                self.results.append(result)

                # Save intermediate results
                self.save_results()

                print(f"\n✅ Completed N={n}")

            except Exception as e:
                print(f"\n❌ Error for N={n}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ All experiments completed!")
        print(f"📊 Results saved in: {self.results_dir}")

def main():
    parser = argparse.ArgumentParser(description="Performance Model Validation")
    parser.add_argument('--domain-sizes', type=str,
                        default="1024,2048,4096,8192,16384,32768,51200,65536,81920,102400",
                        help="Comma-separated list of domain sizes")
    parser.add_argument('--output-dir', type=str, default="results/saturation_study",
                        help="Output directory for results")
    parser.add_argument('--base-dir', type=str, default=".",
                        help="Base directory of FRUGAL project")
    parser.add_argument('--num-runs', type=int, default=10,
                        help="Number of runs for averaging (default: 10, use 1 for quick test)")
    parser.add_argument('--mode', type=str, default="all", choices=["all", "baseline-only", "optimized-only"],
                        help="Run mode: all (default), baseline-only, or optimized-only")

    args = parser.parse_args()

    # Parse domain sizes
    domain_sizes = [int(x.strip()) for x in args.domain_sizes.split(',')]

    # Create validator
    validator = PerformanceValidator(
        base_dir=args.base_dir,
        results_dir=args.output_dir,
        num_runs=args.num_runs
    )

    # Run validation based on mode
    if args.mode == "all":
        validator.run_all(domain_sizes)
    elif args.mode == "baseline-only":
        validator.run_baseline_only(domain_sizes)
    elif args.mode == "optimized-only":
        validator.run_optimized_only(domain_sizes)

if __name__ == "__main__":
    main()
