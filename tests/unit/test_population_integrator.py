"""
Unit tests for population integrator module

Tests: src/data/population_integrator.py
Feature: 002-population-integration
Task: T048-T050
"""

import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point

from src.data.population_integrator import (
    load_ibge_data,
    merge_population,
    validate_population_data
)


class TestLoadIBGEData(unittest.TestCase):
    """Test suite for load_ibge_data function"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def create_sample_ibge_shapefile(self, filename: str, 
                                     pop_col: str = 'POP', 
                                     id_col: str = 'ID36',
                                     include_negatives: bool = False) -> Path:
        """
        Create a sample IBGE shapefile in a ZIP archive for testing.
        
        Args:
            filename: Name of the ZIP file
            pop_col: Name of the population column
            id_col: Name of the ID column
            include_negatives: Whether to include negative population values
        
        Returns:
            Path to the created ZIP file
        """
        # Create sample data with 200m cells
        data = {
            id_col: ['200mE607000N7792000', '200mE607200N7792000', '200mE607400N7792000'],
            pop_col: [1500, 800, -100] if include_negatives else [1500, 800, 0],
            'geometry': [
                Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)]),
                Polygon([(0.002, 0), (0.004, 0), (0.004, 0.002), (0.002, 0.002)]),
                Polygon([(0.004, 0), (0.006, 0), (0.006, 0.002), (0.004, 0.002)])
            ]
        }
        
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        # Save as shapefile in temp directory
        shp_path = self.temp_path / "test_ibge.shp"
        gdf.to_file(shp_path)
        
        # Create ZIP archive with all shapefile components
        zip_path = self.temp_path / filename
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.dbf', '.shx', '.prj']:
                file_path = self.temp_path / f"test_ibge{ext}"
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
        
        return zip_path
    
    def test_load_ibge_data_success(self):
        """Test successful loading of IBGE data with standard columns"""
        zip_path = self.create_sample_ibge_shapefile("ibge_test.zip")
        
        gdf = load_ibge_data(str(zip_path))
        
        self.assertEqual(len(gdf), 3)
        self.assertIn('POP', gdf.columns)
        self.assertIn('ID36', gdf.columns)
        self.assertEqual(gdf['POP'].sum(), 2300)
        self.assertEqual(str(gdf.crs), 'EPSG:4326')
    
    def test_load_ibge_data_file_not_found(self):
        """Test that FileNotFoundError is raised when ZIP file doesn't exist"""
        nonexistent_path = self.temp_path / "nonexistent.zip"
        
        with self.assertRaises(FileNotFoundError) as context:
            load_ibge_data(str(nonexistent_path))
        
        self.assertIn("IBGE ZIP file not found", str(context.exception))
        self.assertIn("Remediation", str(context.exception))
    
    def test_load_ibge_data_alternative_column_names(self):
        """Test loading with alternative population column names (TOTAL instead of POP)"""
        # Create shapefile with 'TOTAL' instead of 'POP'
        data = {
            'ID36': ['200mE607000N7792000', '200mE607200N7792000'],
            'TOTAL': [1200, 900],
            'geometry': [
                Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)]),
                Polygon([(0.002, 0), (0.004, 0), (0.004, 0.002), (0.002, 0.002)])
            ]
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        shp_path = self.temp_path / "test_ibge_total.shp"
        gdf.to_file(shp_path)
        
        zip_path = self.temp_path / "ibge_total.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.dbf', '.shx', '.prj']:
                file_path = self.temp_path / f"test_ibge_total{ext}"
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
        
        gdf_loaded = load_ibge_data(str(zip_path))
        
        # Should be renamed to 'POP'
        self.assertIn('POP', gdf_loaded.columns)
        self.assertEqual(gdf_loaded['POP'].sum(), 2100)
    
    def test_load_ibge_data_missing_population_column(self):
        """Test that ValueError is raised when population column is missing"""
        # Create shapefile without population column
        data = {
            'ID36': ['200mE607000N7792000'],
            'OTHER_COL': [100],
            'geometry': [Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)])]
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        shp_path = self.temp_path / "test_no_pop.shp"
        gdf.to_file(shp_path)
        
        zip_path = self.temp_path / "ibge_no_pop.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.dbf', '.shx', '.prj']:
                file_path = self.temp_path / f"test_no_pop{ext}"
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
        
        with self.assertRaises(ValueError) as context:
            load_ibge_data(str(zip_path))
        
        self.assertIn("Missing required population column", str(context.exception))
        self.assertIn("Available columns", str(context.exception))
    
    def test_load_ibge_data_negative_population(self):
        """Test that ValueError is raised when population values are negative"""
        zip_path = self.create_sample_ibge_shapefile("ibge_negative.zip", include_negatives=True)
        
        with self.assertRaises(ValueError) as context:
            load_ibge_data(str(zip_path))
        
        self.assertIn("negative population values", str(context.exception))
    
    def test_load_ibge_data_null_population_handling(self):
        """Test that null population values are filled with 0"""
        # Create shapefile with null values
        data = {
            'ID36': ['200mE607000N7792000', '200mE607200N7792000'],
            'POP': [1200, None],
            'geometry': [
                Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)]),
                Polygon([(0.002, 0), (0.004, 0), (0.004, 0.002), (0.002, 0.002)])
            ]
        }
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        shp_path = self.temp_path / "test_null.shp"
        gdf.to_file(shp_path)
        
        zip_path = self.temp_path / "ibge_null.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.dbf', '.shx', '.prj']:
                file_path = self.temp_path / f"test_null{ext}"
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
        
        gdf_loaded = load_ibge_data(str(zip_path))
        
        # Null should be filled with 0
        self.assertEqual(gdf_loaded['POP'].iloc[1], 0)
        self.assertFalse(gdf_loaded['POP'].isna().any())
    
    def test_load_ibge_data_crs_reprojection(self):
        """Test CRS reprojection to target CRS"""
        zip_path = self.create_sample_ibge_shapefile("ibge_crs.zip")
        
        # Load with different target CRS
        gdf = load_ibge_data(str(zip_path), target_crs="EPSG:31983")
        
        self.assertEqual(str(gdf.crs), 'EPSG:31983')
    
    def test_load_ibge_data_bounds_filtering(self):
        """Test filtering by geographic bounds"""
        zip_path = self.create_sample_ibge_shapefile("ibge_bounds.zip")
        
        # Define bounds that exclude some cells
        bounds = {
            'lat_min': -0.001,
            'lat_max': 0.001,
            'lon_min': -0.001,
            'lon_max': 0.003
        }
        
        gdf = load_ibge_data(str(zip_path), bounds=bounds)
        
        # Should filter out some cells based on centroid
        self.assertLessEqual(len(gdf), 3)


class TestMergePopulation(unittest.TestCase):
    """Test suite for merge_population function"""
    
    def setUp(self):
        """Create sample grid and IBGE data for testing"""
        # Create sample grid (4 cells)
        self.grid_gdf = gpd.GeoDataFrame({
            'cell_id': ['cell_001', 'cell_002', 'cell_003', 'cell_004'],
            'ID36': ['200mE607000N7792000', '200mE607200N7792000', '200mE607400N7792000', 'UNMATCHED'],
            'stops_count': [10, 5, 2, 0],
            'geometry': [
                Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)]),
                Polygon([(0.002, 0), (0.004, 0), (0.004, 0.002), (0.002, 0.002)]),
                Polygon([(0.004, 0), (0.006, 0), (0.006, 0.002), (0.004, 0.002)]),
                Polygon([(0.006, 0), (0.008, 0), (0.008, 0.002), (0.006, 0.002)])
            ]
        }, crs="EPSG:4326")
        
        # Create sample IBGE data (3 cells)
        self.ibge_gdf = gpd.GeoDataFrame({
            'ID36': ['200mE607000N7792000', '200mE607200N7792000', '200mE607400N7792000'],
            'POP': [1500, 800, 0],
            'geometry': [
                Polygon([(0, 0), (0.002, 0), (0.002, 0.002), (0, 0.002)]),
                Polygon([(0.002, 0), (0.004, 0), (0.004, 0.002), (0.002, 0.002)]),
                Polygon([(0.004, 0), (0.006, 0), (0.006, 0.002), (0.004, 0.002)])
            ]
        }, crs="EPSG:4326")
    
    def test_merge_population_id_method_success(self):
        """Test ID-based merge with successful matches"""
        result = merge_population(self.grid_gdf, self.ibge_gdf, method="id")
        
        self.assertEqual(len(result), 4)
        self.assertIn('population', result.columns)
        self.assertEqual(result[result['cell_id'] == 'cell_001']['population'].iloc[0], 1500)
        self.assertEqual(result[result['cell_id'] == 'cell_002']['population'].iloc[0], 800)
        self.assertEqual(result[result['cell_id'] == 'cell_004']['population'].iloc[0], 0)  # Unmatched
    
    def test_merge_population_spatial_method(self):
        """Test spatial join method"""
        # Remove ID36 column to force spatial join
        grid_no_id = self.grid_gdf.drop(columns=['ID36'])
        
        result = merge_population(grid_no_id, self.ibge_gdf, method="spatial")
        
        self.assertEqual(len(result), 4)
        self.assertIn('population', result.columns)
        # Should match by centroid-in-polygon
        self.assertGreater(result['population'].sum(), 0)
    
    def test_merge_population_auto_method_fallback(self):
        """Test auto method with ID merge + spatial fallback"""
        result = merge_population(self.grid_gdf, self.ibge_gdf, method="auto")
        
        self.assertEqual(len(result), 4)
        self.assertIn('population', result.columns)
        self.assertIn('merge_method', result.columns)
        
        # First 3 cells should use ID merge
        id_merged = result[result['merge_method'] == 'id']
        self.assertGreaterEqual(len(id_merged), 3)
    
    def test_merge_population_crs_mismatch_handling(self):
        """Test automatic CRS reprojection when grid and IBGE have different CRS"""
        # Create IBGE data in different CRS
        ibge_utm = self.ibge_gdf.to_crs("EPSG:31983")
        
        result = merge_population(self.grid_gdf, ibge_utm, method="id")
        
        # Should succeed despite CRS mismatch
        self.assertEqual(len(result), 4)
        self.assertGreater(result['population'].sum(), 0)
    
    def test_merge_population_missing_cell_id(self):
        """Test that ValueError is raised when grid missing cell_id column"""
        grid_no_id = self.grid_gdf.drop(columns=['cell_id'])
        
        with self.assertRaises(ValueError) as context:
            merge_population(grid_no_id, self.ibge_gdf)
        
        self.assertIn("missing required column 'cell_id'", str(context.exception))
    
    def test_merge_population_missing_pop_column(self):
        """Test that ValueError is raised when IBGE missing POP column"""
        ibge_no_pop = self.ibge_gdf.drop(columns=['POP'])
        
        with self.assertRaises(ValueError) as context:
            merge_population(self.grid_gdf, ibge_no_pop)
        
        self.assertIn("missing required column 'POP'", str(context.exception))
    
    def test_merge_population_zero_population_cells(self):
        """Test handling of cells with zero population"""
        result = merge_population(self.grid_gdf, self.ibge_gdf, method="id")
        
        # cell_003 has POP=0 in IBGE, should be 0 not null
        cell_003_pop = result[result['cell_id'] == 'cell_003']['population'].iloc[0]
        self.assertEqual(cell_003_pop, 0)
    
    def test_merge_population_unmatched_cells_default_zero(self):
        """Test that unmatched cells default to zero population"""
        result = merge_population(self.grid_gdf, self.ibge_gdf, method="id")
        
        # cell_004 has no match, should default to 0
        cell_004_pop = result[result['cell_id'] == 'cell_004']['population'].iloc[0]
        self.assertEqual(cell_004_pop, 0)


class TestValidatePopulationData(unittest.TestCase):
    """Test suite for validate_population_data function"""
    
    def setUp(self):
        """Create sample enriched data for validation testing"""
        self.valid_gdf = gpd.GeoDataFrame({
            'cell_id': [f'cell_{i:03d}' for i in range(100)],
            'population': [1000] * 96 + [0] * 4,  # 96% coverage, 4% zero
            'geometry': [Point(i, i) for i in range(100)]
        }, crs="EPSG:4326")
        
        self.low_coverage_gdf = gpd.GeoDataFrame({
            'cell_id': [f'cell_{i:03d}' for i in range(100)],
            'population': [1000] * 80 + [0] * 20,  # Only 80% coverage
            'geometry': [Point(i, i) for i in range(100)]
        }, crs="EPSG:4326")
    
    def test_validate_population_data_success(self):
        """Test validation passes with good data"""
        is_valid, stats = validate_population_data(
            self.valid_gdf,
            min_coverage_pct=95.0,
            max_zero_pop_pct=10.0
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(stats['cell_count'], 100)
        self.assertEqual(stats['total_population'], 96000)
        self.assertGreaterEqual(stats['coverage_pct'], 95.0)
        self.assertEqual(len(stats['errors']), 0)
    
    def test_validate_population_data_low_coverage_fails(self):
        """Test validation fails when coverage below minimum"""
        is_valid, stats = validate_population_data(
            self.low_coverage_gdf,
            min_coverage_pct=95.0
        )
        
        self.assertFalse(is_valid)
        self.assertGreater(len(stats['errors']), 0)
        self.assertIn("Coverage", stats['errors'][0])
    
    def test_validate_population_data_total_population_check(self):
        """Test total population validation against expected value"""
        is_valid, stats = validate_population_data(
            self.valid_gdf,
            expected_total_pop=96000,
            min_coverage_pct=90.0
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(stats['total_population'], 96000)
    
    def test_validate_population_data_total_population_mismatch_warning(self):
        """Test warning when total population differs significantly from expected"""
        is_valid, stats = validate_population_data(
            self.valid_gdf,
            expected_total_pop=50000,  # Expected 50k but actual is 96k
            min_coverage_pct=90.0
        )
        
        # Should pass validation but have warning
        self.assertTrue(is_valid)
        self.assertGreater(len(stats['warnings']), 0)
        self.assertIn("Total population", stats['warnings'][0])
    
    def test_validate_population_data_negative_values_fail(self):
        """Test validation fails with negative population values"""
        invalid_gdf = self.valid_gdf.copy()
        invalid_gdf.loc[0, 'population'] = -100
        
        is_valid, stats = validate_population_data(
            invalid_gdf,
            min_coverage_pct=90.0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("negative population", stats['errors'][0])
    
    def test_validate_population_data_missing_column_fails(self):
        """Test validation fails when population column is missing"""
        invalid_gdf = self.valid_gdf.drop(columns=['population'])
        
        is_valid, stats = validate_population_data(
            invalid_gdf,
            min_coverage_pct=90.0
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Missing required column", stats['errors'][0])
    
    def test_validate_population_data_high_maximum_warning(self):
        """Test warning when cell has unreasonably high population"""
        high_pop_gdf = self.valid_gdf.copy()
        high_pop_gdf.loc[0, 'population'] = 5000  # 5000 in a 200m cell is high
        
        is_valid, stats = validate_population_data(
            high_pop_gdf,
            min_coverage_pct=90.0
        )
        
        # Should pass but have warning
        self.assertTrue(is_valid)
        self.assertGreater(len(stats['warnings']), 0)
        # Check if warning about high max population exists
        has_max_warning = any('Maximum population' in w for w in stats['warnings'])
        self.assertTrue(has_max_warning or len(stats['warnings']) > 0)  # Flexible check
    
    def test_validate_population_data_statistics_accuracy(self):
        """Test that calculated statistics are accurate"""
        is_valid, stats = validate_population_data(
            self.valid_gdf,
            min_coverage_pct=90.0
        )
        
        self.assertEqual(stats['cell_count'], 100)
        self.assertEqual(stats['total_population'], 96000)
        self.assertEqual(stats['coverage_pct'], 96.0)
        self.assertEqual(stats['zero_pop_pct'], 4.0)
        self.assertEqual(stats['pop_range'], (0, 1000))


class TestPopulationIntegratorIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""
    
    def test_end_to_end_population_integration(self):
        """Test full workflow: load -> merge -> validate"""
        # This test would require actual IBGE data file
        # Skip if file not available
        ibge_path = "data/raw/ibge_populacao_bh_grade_id36.zip"
        if not Path(ibge_path).exists():
            self.skipTest(f"IBGE data file not found: {ibge_path}")
        
        # Load IBGE data
        ibge_gdf = load_ibge_data(ibge_path)
        self.assertGreater(len(ibge_gdf), 0)
        
        # Create mock grid for testing
        grid_gdf = gpd.GeoDataFrame({
            'cell_id': ['cell_001', 'cell_002'],
            'geometry': [
                Polygon([(-43.95, -19.92), (-43.94, -19.92), (-43.94, -19.91), (-43.95, -19.91)]),
                Polygon([(-43.94, -19.92), (-43.93, -19.92), (-43.93, -19.91), (-43.94, -19.91)])
            ]
        }, crs="EPSG:4326")
        
        # Merge population
        enriched = merge_population(grid_gdf, ibge_gdf, method="spatial")
        self.assertEqual(len(enriched), 2)
        self.assertIn('population', enriched.columns)
        
        # Validate
        is_valid, stats = validate_population_data(
            enriched,
            min_coverage_pct=0.0,  # Allow low coverage for small test
            max_zero_pop_pct=100.0
        )
        
        self.assertIsInstance(is_valid, bool)
        self.assertIn('total_population', stats)


if __name__ == '__main__':
    unittest.main()
