#!/usr/bin/env python3
"""
Results Organizer - Clean organization and comparison of benchmark results

Organizes results by date, model, benchmark for easy comparison and analysis.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Note: pandas removed to avoid dependency issues
# Can be added later for advanced analysis if needed

class ResultsOrganizer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.organized_dir = self.results_dir / "organized"
        self.comparisons_dir = self.results_dir / "comparisons"
        self.archive_dir = self.results_dir / "archive"
        
        # Create organized structure
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create organized results directory structure"""
        directories = [
            self.organized_dir,
            self.organized_dir / "by_date",
            self.organized_dir / "by_model", 
            self.organized_dir / "by_benchmark",
            self.organized_dir / "suites",
            self.comparisons_dir,
            self.comparisons_dir / "model_vs_model",
            self.comparisons_dir / "benchmark_vs_benchmark", 
            self.comparisons_dir / "time_series",
            self.archive_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def organize_existing_results(self):
        """Organize existing results from the flat structure"""
        print("📁 Organizing existing results...")
        
        organized_count = 0
        
        # Process all JSON files in results directory
        for file_path in self.results_dir.glob("*.json"):
            if self._is_organized_file(file_path):
                continue
                
            result = self._organize_single_file(file_path)
            if result:
                organized_count += 1
                
        print(f"✅ Organized {organized_count} result files")
        return organized_count
        
    def _is_organized_file(self, file_path: Path) -> bool:
        """Check if file is already in organized structure"""
        return any(parent.name in ['organized', 'comparisons', 'archive'] 
                  for parent in file_path.parents)
        
    def _organize_single_file(self, file_path: Path) -> bool:
        """Organize a single result file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Determine file type and extract metadata
            file_info = self._extract_file_metadata(file_path, data)
            if not file_info:
                return False
                
            # Create organized paths
            organized_paths = self._create_organized_paths(file_info)
            
            # Copy file to organized locations
            for org_path in organized_paths:
                org_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, org_path)
                
            # Move original to archive
            archive_path = self.archive_dir / file_path.name
            shutil.move(file_path, archive_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error organizing {file_path}: {e}")
            return False
            
    def _extract_file_metadata(self, file_path: Path, data: Dict) -> Optional[Dict]:
        """Extract metadata from result file"""
        file_name = file_path.stem
        
        # Suite files
        if file_name.startswith('suite_'):
            return {
                'type': 'suite',
                'suite_name': data.get('suite', 'unknown'),
                'models': data.get('models', []),
                'date': self._parse_date_from_data(data),
                'timestamp': self._extract_timestamp_from_filename(file_name),
                'filename': file_path.name
            }
            
        # Metrics files  
        elif file_name.startswith('metrics_'):
            config = data.get('config', {})
            return {
                'type': 'metrics',
                'model': self._clean_model_name(config.get('model', 'unknown')),
                'benchmark': config.get('tasks', 'unknown'),
                'date': self._parse_date_from_timestamp(self._extract_timestamp_from_filename(file_name)),
                'timestamp': self._extract_timestamp_from_filename(file_name),
                'filename': file_path.name,
                'scores': {k: v for k, v in data.items() if k != 'config'}
            }
            
        # Generation files
        elif file_name.startswith('generations_'):
            # Extract info from filename pattern: generations_timestamp_benchmark.json
            parts = file_name.split('_')
            if len(parts) >= 3:
                return {
                    'type': 'generations', 
                    'benchmark': parts[-1],  # Last part before .json
                    'timestamp': parts[1],   # Middle timestamp
                    'date': self._parse_date_from_timestamp(parts[1]),
                    'filename': file_path.name
                }
                
        return None
        
    def _create_organized_paths(self, file_info: Dict) -> List[Path]:
        """Create all organized path locations for a file"""
        paths = []
        filename = file_info['filename']
        date_str = file_info['date'].strftime('%Y-%m-%d') if file_info.get('date') else 'unknown'
        
        if file_info['type'] == 'suite':
            # Suites go in suites directory
            suite_path = self.organized_dir / "suites" / f"{file_info['suite_name']}_{date_str}_{filename}"
            paths.append(suite_path)
            
            # Also by date
            date_path = self.organized_dir / "by_date" / date_str / filename
            paths.append(date_path)
            
        elif file_info['type'] == 'metrics':
            model = file_info['model']
            benchmark = file_info['benchmark']
            
            # By model
            model_path = self.organized_dir / "by_model" / model / f"{benchmark}_{date_str}_{filename}"
            paths.append(model_path)
            
            # By benchmark  
            bench_path = self.organized_dir / "by_benchmark" / benchmark / f"{model}_{date_str}_{filename}"
            paths.append(bench_path)
            
            # By date
            date_path = self.organized_dir / "by_date" / date_str / filename
            paths.append(date_path)
            
        elif file_info['type'] == 'generations':
            # Generations go with their corresponding metrics
            gen_path = self.organized_dir / "by_date" / date_str / filename
            paths.append(gen_path)
            
        return paths
        
    def create_model_comparison(self, models: List[str], benchmark: str = None) -> str:
        """Create a comparison report between models"""
        print(f"📊 Creating model comparison: {', '.join(models)}")
        
        comparison_data = {
            'models': models,
            'benchmark_filter': benchmark,
            'generated_at': datetime.now().isoformat(),
            'comparisons': {}
        }
        
        # Collect results for each model
        model_results = {}
        for model in models:
            model_dir = self.organized_dir / "by_model" / model
            if model_dir.exists():
                model_results[model] = self._collect_model_results(model_dir, benchmark)
                
        # Create comparison tables
        comparison_data['comparisons'] = self._generate_comparison_tables(model_results)
        
        # Save comparison
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = self.comparisons_dir / "model_vs_model" / f"comparison_{'-vs-'.join(models)}_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
            
        print(f"✅ Model comparison saved: {comparison_file}")
        return str(comparison_file)
        
    def create_leaderboard(self, benchmark: str = "humaneval") -> str:
        """Create a leaderboard for a specific benchmark"""
        print(f"🏆 Creating leaderboard for {benchmark}")
        
        benchmark_dir = self.organized_dir / "by_benchmark" / benchmark
        if not benchmark_dir.exists():
            print(f"❌ No results found for benchmark: {benchmark}")
            return None
            
        leaderboard_data = {
            'benchmark': benchmark,
            'generated_at': datetime.now().isoformat(),
            'rankings': []
        }
        
        # Collect all results for this benchmark
        all_results = []
        for result_file in benchmark_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                config = data.get('config', {})
                scores = {k: v for k, v in data.items() if k != 'config'}
                
                # Extract primary score (pass@1 for most benchmarks)
                primary_score = self._extract_primary_score(scores, benchmark)
                
                result_entry = {
                    'model': self._clean_model_name(config.get('model', 'unknown')),
                    'primary_score': primary_score,
                    'all_scores': scores,
                    'config': config,
                    'file': result_file.name
                }
                all_results.append(result_entry)
                
            except Exception as e:
                print(f"❌ Error processing {result_file}: {e}")
                
        # Sort by primary score (descending)
        all_results.sort(key=lambda x: x['primary_score'], reverse=True)
        leaderboard_data['rankings'] = all_results
        
        # Save leaderboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') 
        leaderboard_file = self.comparisons_dir / f"leaderboard_{benchmark}_{timestamp}.json"
        
        with open(leaderboard_file, 'w') as f:
            json.dump(leaderboard_data, f, indent=2)
            
        print(f"✅ Leaderboard saved: {leaderboard_file}")
        return str(leaderboard_file)
        
    def _collect_model_results(self, model_dir: Path, benchmark_filter: str = None) -> Dict:
        """Collect all results for a model"""
        results = {}
        
        for result_file in model_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                config = data.get('config', {})
                benchmark = config.get('tasks', 'unknown')
                
                if benchmark_filter and benchmark != benchmark_filter:
                    continue
                    
                scores = {k: v for k, v in data.items() if k != 'config'}
                results[benchmark] = {
                    'scores': scores,
                    'config': config,
                    'file': result_file.name
                }
                
            except Exception as e:
                print(f"❌ Error reading {result_file}: {e}")
                
        return results
        
    def _generate_comparison_tables(self, model_results: Dict) -> Dict:
        """Generate comparison tables from model results"""
        comparisons = {}
        
        # Find common benchmarks
        all_benchmarks = set()
        for model_data in model_results.values():
            all_benchmarks.update(model_data.keys())
            
        # Create comparison for each benchmark
        for benchmark in all_benchmarks:
            comparison_table = []
            
            for model, results in model_results.items():
                if benchmark in results:
                    scores = results[benchmark]['scores']
                    primary_score = self._extract_primary_score(scores, benchmark)
                    
                    comparison_table.append({
                        'model': model,
                        'primary_score': primary_score,
                        'all_scores': scores
                    })
                    
            # Sort by primary score
            comparison_table.sort(key=lambda x: x['primary_score'], reverse=True)
            comparisons[benchmark] = comparison_table
            
        return comparisons
        
    def _extract_primary_score(self, scores: Dict, benchmark: str) -> float:
        """Extract the primary score for a benchmark"""
        benchmark_lower = benchmark.lower()
        
        # Most benchmarks use pass@1
        if benchmark_lower in scores:
            benchmark_scores = scores[benchmark_lower]
            if isinstance(benchmark_scores, dict) and 'pass@1' in benchmark_scores:
                return float(benchmark_scores['pass@1'])
                
        # Fallback to first numeric value found
        for value in scores.values():
            if isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, (int, float)):
                        return float(sub_value)
            elif isinstance(value, (int, float)):
                return float(value)
                
        return 0.0
        
    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name for file organization"""
        return model_name.replace('/', '_').replace(' ', '_')
        
    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from filename"""
        parts = filename.split('_')
        for part in parts:
            if part.isdigit() and len(part) == 10:  # Unix timestamp
                return part
        return "0"
        
    def _parse_date_from_timestamp(self, timestamp: str) -> datetime:
        """Parse date from unix timestamp"""
        try:
            return datetime.fromtimestamp(int(timestamp))
        except:
            return datetime.now()
            
    def _parse_date_from_data(self, data: Dict) -> datetime:
        """Parse date from result data"""
        if 'start_time' in data:
            try:
                return datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
            except:
                pass
                
        return datetime.now()

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize and compare benchmark results")
    parser.add_argument('--organize', action='store_true', help='Organize existing results')
    parser.add_argument('--compare', nargs='+', help='Compare models')
    parser.add_argument('--leaderboard', help='Create leaderboard for benchmark')
    parser.add_argument('--benchmark', help='Filter by benchmark for comparisons')
    
    args = parser.parse_args()
    
    organizer = ResultsOrganizer()
    
    if args.organize:
        organizer.organize_existing_results()
        
    if args.compare:
        organizer.create_model_comparison(args.compare, args.benchmark)
        
    if args.leaderboard:
        organizer.create_leaderboard(args.leaderboard)
        
if __name__ == "__main__":
    main()