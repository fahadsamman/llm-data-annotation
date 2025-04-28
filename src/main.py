import os
import argparse
from src.data_loader import load_and_validate_data, DataLoaderError


def main():

    print("Hello, world!")

    # === Data loading ===

    parser = argparse.ArgumentParser(description="Load dataset with validation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()

    try:
        df = load_and_validate_data(args.data_path)
        print(f"Dataset loaded successfully with shape {df.shape}")
    except DataLoaderError as e:
        print(f"Data loading failed: {e}")



    # === Autogen config ===
    config = {
        "llm_config": {
            "timeout": 60,
            "seed": 42,
            "config_list": [
                {
                    "model": "gpt-4.1-nano",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    #"base_url": "https://api.deepseek.com"
                }
            ],
        }
    }

if __name__ == "__main__":
    main()
