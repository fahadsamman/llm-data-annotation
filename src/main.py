import os
import argparse
from src.data_loader import load_and_validate_data, DataLoaderError
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv

SYSTEM_PROMPT = """
You are a data annotator. Given a dataset, you need to annotate the dataset with the following labels:

yes: the text is about a restaurant
no: the text is not about a restaurant
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Load dataset with validation")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    return parser.parse_args()


def main():


    args = parse_args()

    print("Dataset loaded successfully:", args)


    load_dotenv()
    print("Environment variables loaded successfully.")


    # === Data loading ===
    try:
        df = load_and_validate_data(args.data)
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
                    "model": "deepseek-chat",
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                    "base_url": "https://api.deepseek.com",
                    "price": [0.00027, 0.00110]  # [input_price_per_1k, output_price_per_1k] according to chatgpt...
                }
            ],
        }
    }

    # === Clean Data ===
    df["ai_label"] = ""  # Pre-fill output column 


    # === Agents ===
    surplus_assessor = AssistantAgent(
        name="surplus_assessor",
        system_message=SYSTEM_PROMPT,
        llm_config=config["llm_config"]
    )


    reply = surplus_assessor.generate_reply(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": df["text"][0]},
                        ]
                    )
    
    print(reply)





if __name__ == "__main__":
    main()
