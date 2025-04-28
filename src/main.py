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
    
def generate_system_prompt(batch_size, positive_label, negative_label):
    
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

def save_results(df, prefix="annotated_data"):
    """Save dataframe to results/ folder with a datetime-stamped filename."""

    # Create results folder if it doesn't exist
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate clean timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Build output filename
    output_filename = f"{prefix}_{timestamp}.csv"
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
    if not api_key:
        raise ValueError("API_KEY not found in environment variables.")
    print("✅ API key loaded successfully.")

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

    positive_label = config.get("positive_label")
    negative_label = config.get("negative_label")
    text_column = config["text_column"] if "text_column" in config else "text"
    id_column = config["id_column"] if "id_column" in config else "id"
    validation_column = config["validation_column"] if "validation_column" in config else None

    if args.batch_size is None:
        batch_size = config["batch_size"]
    else:
        batch_size = args.batch_size

    ## get dataset path from config file but allow CLI override:
    if args.data is None:
        dataset_path = config["dataset_path"] 
    elif args.data is not None:
        dataset_path = args.data 
    else:
        raise ValueError("No dataset path specified in config or CLI args.")

    ## if rows_to_label is specified, only select the first rows_to_label rows:
    try:
        rows_to_label = config["rows_to_label"] 
    except KeyError:
        rows_to_label = None

    #results_dir = config["defaults"]["results_dir"]
    #base_filename = config["defaults"]["base_filename"]

    system_prompt = generate_system_prompt(
        batch_size=batch_size, 
        positive_label=positive_label, 
        negative_label=negative_label
        )
    
    print("✅ Config loaded successfully.")


    # === Data loading ===
    try:
        df_raw = load_and_validate_data(dataset_path)
        df_raw['id'] = df_raw[id_column].astype(str)
        df_raw['text'] = df_raw[text_column].astype(str)
        if validation_column is not None:
            df_raw['validation_label'] = df_raw[validation_column].astype(str)
        df_raw["ai_label"] = ""  # Pre-fill output column 

        df = df_raw[["id", "text", "ai_label"]].copy()
        print(f"✅ Dataset loaded successfully with path {dataset_path} and shape {df.shape}")


    except DataLoaderError as e:
        print(f"❌ Data loading failed: {e}")
        
    
    ## if rows_to_label is specified, only select the first rows_to_label rows:
    if rows_to_label is not None:
        df = df.head(rows_to_label)


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
    save_results(df,'annotated_data')

    df_raw[["ai_label"]] = df[["ai_label"]]

    save_results(df_raw,'full_annotated_data') 




if __name__ == "__main__":
    main()
