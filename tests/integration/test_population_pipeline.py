"""
Integration tests for population integration pipeline

Tests: Full pipeline from config → grid → features → population → validation
Feature: 002-population-integration
Tasks: T051-T052
"""

import unittest
import time
from pathlib import Path
import shutil
import yaml

import pandas as pd
import geopandas as gpd

from src.data.grid_generator import GridGenerator
from src.data.feature_extractor import FeatureExtractor
from src.data.population_integrator import (
    load_ibge_data,
    merge_population,
    validate_population_data
)


class TestPopulationPipelineIntegration(unittest.TestCase):
    """Integration test for complete population integration pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment once for all tests"""
        cls.config_path = Path("config/model_config.yaml")
        cls.test_output_dir = Path("data/processed/test_integration")
        cls.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Check if IBGE data is available
        cls.ibge_path = Path("data/raw/ibge_populacao_bh_grade_id36.zip")
        cls.has_ibge_data = cls.ibge_path.exists()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test outputs"""
        if cls.test_output_dir.exists():
            shutil.rmtree(cls.test_output_dir)
    
    def test_01_config_validation(self):
        """Test that configuration is valid for population integration"""
        # Verify grid configuration exists
        self.assertIn('grid', self.config)
        self.assertIn('cell_size_meters', self.config['grid'])
        
        # Verify grid size is 200m (required for IBGE alignment)
        cell_size = self.config['grid']['cell_size_meters']
        self.assertEqual(
            cell_size, 200,
            f"Grid size must be 200m for IBGE alignment, found {cell_size}m"
        )
        
        # Verify bounds are configured
        self.assertIn('bounds', self.config['grid'])
        bounds = self.config['grid']['bounds']
        self.assertIn('min_lat', bounds)
        self.assertIn('max_lat', bounds)
        self.assertIn('min_lon', bounds)
        self.assertIn('max_lon', bounds)
        
        # Verify bounds cover Belo Horizonte
        self.assertGreater(bounds['min_lat'], -20.1)
        self.assertLess(bounds['max_lat'], -19.7)
        self.assertGreater(bounds['min_lon'], -44.1)
        self.assertLess(bounds['max_lon'], -43.8)
    
    def test_02_grid_generation(self):
        """Test grid generation step"""
        generator = GridGenerator(config=self.config)
        
        # Generate grid
        grid_gdf = generator.generate_grid()
        
        # Verify grid properties
        self.assertIsInstance(grid_gdf, gpd.GeoDataFrame)
        self.assertGreater(len(grid_gdf), 0)
        self.assertIn('cell_id', grid_gdf.columns)
        self.assertIn('geometry', grid_gdf.columns)
        
        # Verify CRS
        self.assertIsNotNone(grid_gdf.crs)
        
        # Save for next test
        grid_path = self.test_output_dir / "grid.parquet"
        grid_gdf.to_parquet(grid_path)
        
        print(f"✓ Generated {len(grid_gdf)} grid cells at 200m resolution")
    
    def test_03_feature_extraction(self):
        """Test GTFS feature extraction step"""
        # Load grid from previous test
        grid_path = self.test_output_dir / "grid.parquet"
        if not grid_path.exists():
            self.skipTest("Grid not generated - run test_02_grid_generation first")
        
        grid_gdf = gpd.read_parquet(grid_path)
        
        # Extract features
        extractor = FeatureExtractor(config=self.config)
        features_gdf = extractor.extract_features(grid_gdf)
        
        # Verify feature extraction
        self.assertIsInstance(features_gdf, gpd.GeoDataFrame)
        self.assertEqual(len(features_gdf), len(grid_gdf))
        
        # Check for expected features
        expected_features = ['stops_count', 'routes_count', 'trips_count']
        for feature in expected_features:
            self.assertIn(
                feature, features_gdf.columns,
                f"Missing expected feature: {feature}"
            )
        
        # Save for next test
        features_path = self.test_output_dir / "features.parquet"
        features_gdf.to_parquet(features_path)
        
        print(f"✓ Extracted transit features for {len(features_gdf)} cells")
    
    def test_04_population_integration(self):
        """Test IBGE population integration step"""
        # Load features from previous test
        features_path = self.test_output_dir / "features.parquet"
        if not features_path.exists():
            self.skipTest("Features not extracted - run test_03_feature_extraction first")
        
        features_gdf = gpd.read_parquet(features_path)
        
        # Check IBGE data availability
        if not self.has_ibge_data:
            self.skipTest(f"IBGE data not found at {self.ibge_path}")
        
        # Load IBGE data
        ibge_gdf = load_ibge_data(str(self.ibge_path))
        
        # Verify IBGE data loaded successfully
        self.assertIsInstance(ibge_gdf, gpd.GeoDataFrame)
        self.assertGreater(len(ibge_gdf), 0)
        self.assertIn('POP', ibge_gdf.columns)
        self.assertIn('ID36', ibge_gdf.columns)
        
        # Merge population data
        enriched_gdf = merge_population(
            features_gdf,
            ibge_gdf,
            method="auto"
        )
        
        # Verify merge successful
        self.assertEqual(len(enriched_gdf), len(features_gdf))
        self.assertIn('population', enriched_gdf.columns)
        
        # Check population statistics
        total_pop = enriched_gdf['population'].sum()
        self.assertGreater(total_pop, 0, "Total population should be > 0")
        
        pop_coverage = (enriched_gdf['population'] > 0).sum() / len(enriched_gdf) * 100
        self.assertGreater(
            pop_coverage, 50,
            f"Population coverage too low: {pop_coverage:.1f}%"
        )
        
        # Save for next test
        enriched_path = self.test_output_dir / "features_with_population.parquet"
        enriched_gdf.to_parquet(enriched_path)
        
        print(f"✓ Integrated population data: {total_pop:,.0f} inhabitants")
        print(f"  Coverage: {pop_coverage:.1f}% of cells have population data")
    
    def test_05_population_validation(self):
        """Test population data validation step"""
        # Load enriched features from previous test
        enriched_path = self.test_output_dir / "features_with_population.parquet"
        if not enriched_path.exists():
            self.skipTest("Population not integrated - run test_04_population_integration first")
        
        enriched_gdf = gpd.read_parquet(enriched_path)
        
        # Validate population data
        is_valid, stats = validate_population_data(
            enriched_gdf,
            min_coverage_pct=90.0,  # SC-001: ≥90% IBGE cells merged
            max_zero_pop_pct=15.0,
            expected_total_pop=None  # Don't enforce exact total for test
        )
        
        # Display validation results
        print(f"\n  Validation Results:")
        print(f"    Total cells: {stats['cell_count']}")
        print(f"    Total population: {stats['total_population']:,.0f}")
        print(f"    Coverage: {stats['coverage_pct']:.1f}%")
        print(f"    Zero-pop cells: {stats['zero_pop_pct']:.1f}%")
        print(f"    Pop range: {stats['pop_range'][0]:,.0f} - {stats['pop_range'][1]:,.0f}")
        
        if stats['errors']:
            print(f"    Errors: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"      - {error}")
        
        if stats['warnings']:
            print(f"    Warnings: {len(stats['warnings'])}")
            for warning in stats['warnings']:
                print(f"      - {warning}")
        
        # Verify validation criteria
        self.assertGreaterEqual(
            stats['coverage_pct'], 90.0,
            f"Coverage {stats['coverage_pct']:.1f}% below 90% threshold (SC-001)"
        )
        
        self.assertLessEqual(
            stats['zero_pop_pct'], 15.0,
            f"Zero-pop cells {stats['zero_pop_pct']:.1f}% above 15% threshold"
        )
        
        # Verify total population is reasonable for BH
        # Belo Horizonte: ~2.3-2.7M inhabitants
        self.assertGreater(
            stats['total_population'], 2_000_000,
            f"Total population {stats['total_population']:,.0f} seems too low for BH"
        )
        
        self.assertLess(
            stats['total_population'], 3_000_000,
            f"Total population {stats['total_population']:,.0f} seems too high for BH"
        )
        
        print(f"\n✓ Population validation passed (SC-001 met)")
    
    def test_06_end_to_end_pipeline_success(self):
        """Test complete pipeline execution from start to finish"""
        if not self.has_ibge_data:
            self.skipTest(f"IBGE data not found at {self.ibge_path}")
        
        print("\nRunning end-to-end pipeline test...")
        
        # Step 1: Grid generation
        print("  [1/5] Generating grid...")
        generator = GridGenerator(config=self.config)
        grid_gdf = generator.generate_grid()
        self.assertGreater(len(grid_gdf), 0)
        
        # Step 2: Feature extraction
        print("  [2/5] Extracting transit features...")
        extractor = FeatureExtractor(config=self.config)
        features_gdf = extractor.extract_features(grid_gdf)
        self.assertEqual(len(features_gdf), len(grid_gdf))
        
        # Step 3: Load IBGE data
        print("  [3/5] Loading IBGE population data...")
        ibge_gdf = load_ibge_data(str(self.ibge_path))
        self.assertGreater(len(ibge_gdf), 0)
        
        # Step 4: Merge population
        print("  [4/5] Merging population data...")
        enriched_gdf = merge_population(features_gdf, ibge_gdf, method="auto")
        self.assertEqual(len(enriched_gdf), len(features_gdf))
        self.assertIn('population', enriched_gdf.columns)
        
        # Step 5: Validate
        print("  [5/5] Validating population integration...")
        is_valid, stats = validate_population_data(
            enriched_gdf,
            min_coverage_pct=90.0
        )
        
        # Verify success criteria
        self.assertTrue(is_valid, "Pipeline validation failed")
        self.assertGreaterEqual(
            stats['coverage_pct'], 90.0,
            "Coverage below 90% (SC-001)"
        )
        
        print(f"\n✓ End-to-end pipeline completed successfully")
        print(f"  Grid cells: {len(enriched_gdf)}")
        print(f"  Population: {stats['total_population']:,.0f}")
        print(f"  Coverage: {stats['coverage_pct']:.1f}%")


class TestPopulationPipelinePerformance(unittest.TestCase):
    """Performance tests for pipeline execution time"""
    
    def setUp(self):
        """Check if IBGE data is available"""
        self.ibge_path = Path("data/raw/ibge_populacao_bh_grade_id36.zip")
        self.has_ibge_data = self.ibge_path.exists()
        
        self.config_path = Path("config/model_config.yaml")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def test_population_integration_performance(self):
        """Test that population integration completes within 5 minutes (FR-017)"""
        if not self.has_ibge_data:
            self.skipTest(f"IBGE data not found at {self.ibge_path}")
        
        print("\nRunning population integration performance test...")
        
        # Generate grid
        print("  Generating grid...")
        generator = GridGenerator(config=self.config)
        grid_gdf = generator.generate_grid()
        
        # Extract features
        print("  Extracting features...")
        extractor = FeatureExtractor(config=self.config)
        features_gdf = extractor.extract_features(grid_gdf)
        
        # Time population integration
        print("  Measuring population integration time...")
        start_time = time.time()
        
        # Load IBGE data
        ibge_gdf = load_ibge_data(str(self.ibge_path))
        
        # Merge population
        enriched_gdf = merge_population(features_gdf, ibge_gdf, method="auto")
        
        # Validate
        is_valid, stats = validate_population_data(
            enriched_gdf,
            min_coverage_pct=90.0
        )
        
        population_time = time.time() - start_time
        
        # FR-017: Population integration must complete in < 5 minutes (300 seconds)
        self.assertLess(
            population_time, 300,
            f"Population integration took {population_time:.1f}s, exceeds 5-minute limit (FR-017)"
        )
        
        print(f"\n✓ Population integration completed in {population_time:.1f}s (< 300s required)")
        print(f"  SC-002: Population step within 5-minute limit ✓")
    
    def test_full_pipeline_performance(self):
        """Test that full pipeline completes within 10 minutes (SC-002)"""
        if not self.has_ibge_data:
            self.skipTest(f"IBGE data not found at {self.ibge_path}")
        
        print("\nRunning full pipeline performance test...")
        
        start_time = time.time()
        
        # Execute full pipeline
        print("  [1/5] Grid generation...")
        generator = GridGenerator(config=self.config)
        grid_gdf = generator.generate_grid()
        
        print("  [2/5] Feature extraction...")
        extractor = FeatureExtractor(config=self.config)
        features_gdf = extractor.extract_features(grid_gdf)
        
        print("  [3/5] Loading IBGE data...")
        ibge_gdf = load_ibge_data(str(self.ibge_path))
        
        print("  [4/5] Merging population...")
        enriched_gdf = merge_population(features_gdf, ibge_gdf, method="auto")
        
        print("  [5/5] Validation...")
        is_valid, stats = validate_population_data(
            enriched_gdf,
            min_coverage_pct=90.0
        )
        
        total_time = time.time() - start_time
        
        # SC-002: Total pipeline must complete in < 10 minutes (600 seconds)
        # Note: This test only covers data preparation steps, not model training
        self.assertLess(
            total_time, 600,
            f"Pipeline took {total_time:.1f}s, exceeds 10-minute limit (SC-002)"
        )
        
        print(f"\n✓ Pipeline data preparation completed in {total_time:.1f}s (< 600s required)")
        print(f"  SC-002: Full pipeline within 10-minute limit ✓")


if __name__ == '__main__':
    # Run tests in order
    unittest.main(verbosity=2)
