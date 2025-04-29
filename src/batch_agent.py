from autogen import UserProxyAgent


class HeadlessBatchAgent(UserProxyAgent):
    def __init__(self, df, batch_size, system_prompt, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.batch_size = batch_size
        self.current_index = 0
        self.auto_reply = True  # prevent stop at each round
        self.system_prompt = system_prompt

    def fetch_next_batch(self):
        if self.current_index >= len(self.df):
            return None
        end = min(self.current_index + self.batch_size, len(self.df))
        batch_df = self.df.iloc[self.current_index:end].copy()
        self.current_index = end
        return batch_df

    def store_labels(self, index_list, labels):
        for i, label in zip(index_list, labels):
            self.df.at[i, "ai_label"] = label

    def run_batches(self, assistant):
        while True:
            print(f"⏳ Fetching batch {self.current_index} to {self.current_index + self.batch_size}")
            batch_df = self.fetch_next_batch()
            if batch_df is None:
                break

            input_rows = batch_df["text"].tolist()
            index_list = batch_df.index.tolist()
            rows_text = "\n".join([f"{i+1}. {row}" for i, row in enumerate(input_rows)])

            # === Try twice: once normally, then one retry on fail ===
            for attempt in range(2):
                try:
                    reply = assistant.generate_reply(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": rows_text}
                        ]
                    )
                    output = reply.strip().splitlines()
                    #answers = [line.strip().lower() for line in output if line.strip()]
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

        return self.df