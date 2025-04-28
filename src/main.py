import datetime
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
# Headless proxy agent for automation
class HeadlessBatchAgent(UserProxyAgent):
    def __init__(self, df, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.batch_size = batch_size
        self.current_index = 0
        self.auto_reply = True  # prevent stop at each round

    def fetch_next_batch(self):
        if self.current_index >= len(self.df):
            return None
        end = min(self.current_index + self.batch_size, len(self.df))
        batch_df = self.df.iloc[self.current_index:end].copy()
        self.current_index = end
        return batch_df

    def store_labels(self, index_list, labels):
        for i, label in zip(index_list, labels):
            self.df.at[i, "is_surplus_model"] = label

    def run_batches(self, assistant):
        while True:
            batch_df = self.fetch_next_batch()
            if batch_df is None:
                break

            input_rows = batch_df["ai_label"].tolist()
            index_list = batch_df.index.tolist()
            rows_text = "\n".join([f"{i+1}. {row}" for i, row in enumerate(input_rows)])

            # === Try twice: once normally, then one retry on fail ===
            for attempt in range(2):
                try:
                    reply = assistant.generate_reply(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": rows_text}
                        ]
                    )
                    output = reply.strip().splitlines()
                    answers = [line.strip().lower() for line in output]

                    if len(answers) != len(index_list):
                        raise ValueError(f"Mismatch in length: expected {len(index_list)}, got {len(answers)}")

                    self.store_labels(index_list, answers)
                    print(f"✅ Processed rows {index_list[0]} to {index_list[-1]} (attempt {attempt + 1})")
                    break  # success, exit retry loop

                except Exception as e:
                    if attempt == 0:
                        print(f"⚠️ Retry batch at rows {index_list[0]} to {index_list[-1]} due to: {e}")
                    else:
                        print(f"❌ Final failure at rows {index_list[0]} to {index_list[-1]}: {e}")
                        self.store_labels(index_list, ["error"] * len(index_list))


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


    # === Instantiate and run ===
    batch_agent = HeadlessBatchAgent(
        name="batch_controller",
        df=df,
        batch_size=50,
        code_execution_config={"use_docker": False}
    )


    groupchat = GroupChat(
        agents=[batch_agent, surplus_assessor],
        messages=[],
        max_round=1
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=config["llm_config"])

    # Run the full pipeline automatically
    batch_agent.run_batches(surplus_assessor)

    # save to results folder:
    # Make sure results/ folder exists
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save to a results file with date
    # === Generate clean timestamp ===
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_filename = f"annotated_data_{timestamp}.csv"

    output_path = os.path.join(RESULTS_DIR, output_filename)
    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    main()
