#!/usr/bin/env python3
"""
Script to convert GTFSBHTRANS.zip to Parquet files
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.gtfs_loader import GTFSLoader


def main():
    print("=" * 70)
    print("GTFS BH Trans Data Converter")
    print("Converting txt files to Parquet format")
    print("=" * 70)
    
    try:
        # Initialize loader with absolute paths from project root
        loader = GTFSLoader(
            zip_path=str(project_root / "data/raw/GTFSBHTRANS.zip"),
            output_dir=str(project_root / "data/processed/gtfs")
        )
        
        # List files
        print("\nFiles found in GTFSBHTRANS.zip:")
        for file in loader.list_files():
            print(f"  - {file}")
        
        # Load and convert all files
        print("\n" + "=" * 70)
        loader.load_all_files()
        
        print("\n" + "=" * 70)
        parquet_paths = loader.convert_all_to_parquet()
        
        # Show summary
        print("\n" + "=" * 70)
        print("\nData Summary:")
        summary = loader.get_summary()
        print(summary.to_string(index=False))
        
        # Show file sizes
        print("\n" + "=" * 70)
        print("\nFile Size Comparison (txt vs parquet):")
        sizes = loader.get_file_sizes()
        print(sizes.to_string(index=False))
        
        total_txt = sizes['txt_size_mb'].sum()
        total_parquet = sizes['parquet_size_mb'].sum()
        overall_ratio = total_txt / total_parquet if total_parquet > 0 else 0
        
        print("\n" + "=" * 70)
        print(f"\nTotal txt size: {total_txt:.2f} MB")
        print(f"Total parquet size: {total_parquet:.2f} MB")
        print(f"Overall compression ratio: {overall_ratio:.2f}x")
        print(f"Space saved: {(total_txt - total_parquet):.2f} MB ({((total_txt - total_parquet) / total_txt * 100):.1f}%)")
        
        print("\n" + "=" * 70)
        print("✓ Conversion completed successfully!")
        print("✓ Parquet files saved in: data/processed/gtfs/")
        print("\nYou can now use these files in your notebooks:")
        print("  from src.data.gtfs_loader import GTFSLoader")
        print("  loader = GTFSLoader()")
        print("  df = loader.load_parquet('stop_times')")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
