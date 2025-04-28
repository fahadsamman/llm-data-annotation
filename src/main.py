import datetime
import os
import argparse
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import yaml


from src.data_loader import load_and_validate_data, DataLoaderError
from src.batch_agent import HeadlessBatchAgent



def load_config(path="config.yml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def generate_system_prompt(batch_size, positive_label="yes", negative_label="no"):
    return f"""
You are a data annotator.

You will be given {batch_size} data points as strings.

For each string, classify it with ONLY one label: '{positive_label}' or '{negative_label}'.

Examples:
- Cafe → {positive_label}
- Restaurant → {negative_label}
- Hotel → {negative_label}
- Shisha Cafe → {negative_label}
- Bakery → {positive_label}
- Fast Food → {negative_label}

REPLY WITH EXACTLY {batch_size} LINES, ONE ANSWER PER LINE.
Only output '{positive_label}' or '{negative_label}' for each row, in the same order.
NO numbers.
NO extra text.
""".strip()


def save_results(df, base_filename="annotated_data"):
    """Save dataframe to results/ folder with a datetime-stamped filename."""

    # Create results folder if it doesn't exist
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate clean timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Build output filename
    output_filename = f"{base_filename}_{timestamp}.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    # Save dataframe
    df.to_csv(output_path, index=False)

    print(f"🎉 Saved labeled data to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Load dataset with validation")
    parser.add_argument("--data", type=str, help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing (default: 10)")
    return parser.parse_args()


def main():


    # === Check for optional CLI arguments ===
    args = parse_args()

    # === Get API key from environment variables ===
    load_dotenv()
    api_key = os.getenv("API_KEY")

    print("API key found and loaded successfully.")

    # === Get API keys from environment variables ===
    config = load_config()

    llm_config = {
        "timeout": config["llm_config"]["timeout"],
        "seed": config["llm_config"]["seed"],
        "config_list": [
            {
                "model": config["llm_config"]["model"],
                "api_key": api_key,  # pulled manually
                "base_url": config["llm_config"]["base_url"],
                "price": config["llm_config"]["price"],
            }
        ]
    }

    positive_label = config["positive_label"]
    negative_label = config["negative_label"]

    if args.batch_size is None:
        batch_size = config["batch_size"]
    else:
        batch_size = args.batch_size

    if args.data is None:
        dataset_path = config["dataset_path"]
    else:
        dataset_path = args.data

    #results_dir = config["defaults"]["results_dir"]
    #base_filename = config["defaults"]["base_filename"]

    system_prompt = generate_system_prompt(
        batch_size=batch_size, 
        positive_label=positive_label, 
        negative_label=negative_label
        )
    
    print("Config loaded successfully.")


    # === Data loading ===
    try:
        df = load_and_validate_data(dataset_path)
        print(f"Dataset loaded successfully with path {args.data}")
        df["ai_label"] = ""  # Pre-fill output column 
        print(f"Dataset loaded successfully with shape {df.shape}")
    except DataLoaderError as e:
        print(f"Data loading failed: {e}")



    # === Agents ===
    surplus_assessor = AssistantAgent(
        name="surplus_assessor",
        system_message=system_prompt,
        llm_config=llm_config
    )

    # === Instantiate and run ===
    batch_agent = HeadlessBatchAgent(
        name = "batch_controller",
        df = df,
        batch_size = batch_size,
        system_prompt = system_prompt,
        code_execution_config={"use_docker": False}
    )


    # test, delete later:
    #reply = surplus_assessor.generate_reply(
    #                    messages=[
    #                        {"role": "system", "content": SYSTEM_PROMPT},
    #                        {"role": "user", "content": df["text"][0]},
    #                    ]
    #                )
    
    #print(reply)

    #groupchat = GroupChat(
    #    agents=[batch_agent, surplus_assessor],
    #    messages=[],
    #    max_round=1
    #)

    #manager = GroupChatManager(groupchat=groupchat, llm_config=config["llm_config"])

    # Run the full pipeline:
    batch_agent.run_batches(surplus_assessor)


    # save to results folder:
    save_results(df)




if __name__ == "__main__":
    main()
