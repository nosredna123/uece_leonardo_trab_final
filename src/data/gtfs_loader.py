"""
GTFS Data Loader
Loads GTFSBHTRANS.zip and converts txt files to parquet format
"""
import zipfile
import pandas as pd
from pathlib import Path
from typing import Dict, List
import os


class GTFSLoader:
    """
    Class for loading and converting GTFS data
    """
    
    def __init__(self, zip_path: str = "data/raw/GTFSBHTRANS.zip", 
                 output_dir: str = "data/processed/gtfs"):
        """
        Initialize GTFS Loader
        
        Args:
            zip_path: Path to the GTFS zip file
            output_dir: Directory to save parquet files
        """
        self.zip_path = Path(zip_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        if not self.zip_path.exists():
            raise FileNotFoundError(f"GTFS zip file not found: {self.zip_path}")
    
    def list_files(self) -> List[str]:
        """
        List all txt files in the GTFS zip
        
        Returns:
            List of txt filenames
        """
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            return [f for f in zip_ref.namelist() if f.endswith('.txt')]
    
    def load_txt_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single txt file from the zip
        
        Args:
            filename: Name of the txt file to load
            
        Returns:
            DataFrame with the file contents
        """
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open(filename) as file:
                df = pd.read_csv(file)
        return df
    
    def convert_to_parquet(self, filename: str, df: pd.DataFrame = None) -> str:
        """
        Convert a txt file to parquet format
        
        Args:
            filename: Name of the txt file
            df: DataFrame to save (if None, will load from zip)
            
        Returns:
            Path to the created parquet file
        """
        if df is None:
            df = self.load_txt_file(filename)
        
        # Create parquet filename
        base_name = Path(filename).stem
        parquet_path = self.output_dir / f"{base_name}.parquet"
        
        # Save as parquet
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        
        print(f"✓ Converted {filename} -> {parquet_path.name} ({len(df)} rows, {len(df.columns)} columns)")
        
        return str(parquet_path)
    
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all txt files from the GTFS zip
        
        Returns:
            Dictionary with filename (without .txt) as key and DataFrame as value
        """
        txt_files = self.list_files()
        
        print(f"Loading {len(txt_files)} GTFS files from {self.zip_path.name}...")
        print("-" * 70)
        
        for filename in txt_files:
            try:
                df = self.load_txt_file(filename)
                base_name = Path(filename).stem
                self.dataframes[base_name] = df
                print(f"✓ Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        
        print("-" * 70)
        print(f"Successfully loaded {len(self.dataframes)} files")
        
        return self.dataframes
    
    def convert_all_to_parquet(self) -> Dict[str, str]:
        """
        Convert all txt files in the zip to parquet format
        
        Returns:
            Dictionary with filename and path to parquet file
        """
        if not self.dataframes:
            self.load_all_files()
        
        print(f"\nConverting to Parquet format...")
        print("-" * 70)
        
        parquet_paths = {}
        
        for name, df in self.dataframes.items():
            try:
                parquet_path = self.convert_to_parquet(f"{name}.txt", df)
                parquet_paths[name] = parquet_path
            except Exception as e:
                print(f"✗ Error converting {name}: {e}")
        
        print("-" * 70)
        print(f"Successfully converted {len(parquet_paths)} files to Parquet")
        print(f"Output directory: {self.output_dir}")
        
        return parquet_paths
    
    def load_parquet(self, table_name: str) -> pd.DataFrame:
        """
        Load a parquet file
        
        Args:
            table_name: Name of the GTFS table (without extension)
            
        Returns:
            DataFrame with the parquet data
        """
        parquet_path = self.output_dir / f"{table_name}.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        return pd.read_parquet(parquet_path)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all loaded dataframes
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.dataframes:
            self.load_all_files()
        
        summary = []
        for name, df in self.dataframes.items():
            summary.append({
                'table': name,
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'column_list': ', '.join(df.columns.tolist())
            })
        
        return pd.DataFrame(summary).sort_values('rows', ascending=False)
    
    def get_file_sizes(self) -> pd.DataFrame:
        """
        Get file sizes comparison between txt and parquet
        
        Returns:
            DataFrame with file size comparison
        """
        sizes = []
        
        # Get txt sizes from zip
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for info in zip_ref.infolist():
                if info.filename.endswith('.txt'):
                    base_name = Path(info.filename).stem
                    parquet_path = self.output_dir / f"{base_name}.parquet"
                    
                    txt_size_mb = info.file_size / (1024 * 1024)
                    parquet_size_mb = 0
                    
                    if parquet_path.exists():
                        parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
                    
                    sizes.append({
                        'file': base_name,
                        'txt_size_mb': round(txt_size_mb, 2),
                        'parquet_size_mb': round(parquet_size_mb, 2),
                        'compression_ratio': round(txt_size_mb / parquet_size_mb, 2) if parquet_size_mb > 0 else 0
                    })
        
        return pd.DataFrame(sizes).sort_values('txt_size_mb', ascending=False)


if __name__ == "__main__":
    # Example usage
    loader = GTFSLoader()
    
    # List files
    print("Files in GTFSBHTRANS.zip:")
    for file in loader.list_files():
        print(f"  - {file}")
    
    # Load and convert all files
    print("\n" + "=" * 70)
    loader.load_all_files()
    
    print("\n" + "=" * 70)
    loader.convert_all_to_parquet()
    
    # Show summary
    print("\n" + "=" * 70)
    print("\nData Summary:")
    print(loader.get_summary().to_string(index=False))
    
    # Show file sizes
    print("\n" + "=" * 70)
    print("\nFile Size Comparison:")
    print(loader.get_file_sizes().to_string(index=False))
