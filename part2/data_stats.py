from transformers import T5TokenizerFast

# The checkpoint required by the PDF
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f]


for split in ["train", "dev"]:
    nl = load_lines(f"data/{split}.nl")
    sql = load_lines(f"data/{split}.sql")

    # Tokenize every question and every SQL query (list of lists of token ids)
    nl_ids = [tokenizer(q).input_ids for q in nl]
    sql_ids = [tokenizer(q).input_ids for q in sql]

    # One line = one (question, SQL) example
    num_examples = len(nl)
    mean_nl_len = sum(len(ids) for ids in nl_ids) / num_examples
    mean_sql_len = sum(len(ids) for ids in sql_ids) / num_examples

    # Vocabulary size = number of DISTINCT token ids that appear in this split
    nl_vocab = len(set(t for ids in nl_ids for t in ids))
    sql_vocab = len(set(t for ids in sql_ids for t in ids))

    print(f"{split}: examples={num_examples}, mean_nl_len={mean_nl_len:.2f}, "
          f"mean_sql_len={mean_sql_len:.2f}, nl_vocab={nl_vocab}, sql_vocab={sql_vocab}")
    
    
# Check 1: longest SQL query (decides max generation length)
train_sql = load_lines("data/train.sql")
train_sql_ids = [tokenizer(q).input_ids for q in train_sql]
print("max SQL length:", max(len(ids) for ids in train_sql_ids))

# # Check 2: can every SQL query be recovered exactly after tokenize -> decode?
# bad = []
# for q, ids in zip(train_sql, train_sql_ids):
#     decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     if decoded != q:
#         bad.append((q, decoded))
# print(f"{len(bad)}/{len(train_sql)} SQL queries are NOT recovered exactly")
# if bad:
#     print("original:", bad[0][0][-120:])
#     print("decoded :", bad[0][1][-120:])

# Table 2: statistics after pre-processing (task prefix + "<" -> "lessthan")
TASK_PREFIX = "translate English to SQL: "

def preprocess_sql(sql):
    return sql.replace("<", "lessthan")

print("\nAfter pre-processing:")
for split in ["train", "dev"]:
    nl = load_lines(f"data/{split}.nl")
    sql = load_lines(f"data/{split}.sql")

    nl_ids = [tokenizer(TASK_PREFIX + q).input_ids for q in nl]
    sql_ids = [tokenizer(preprocess_sql(q)).input_ids for q in sql]

    mean_nl_len = sum(len(ids) for ids in nl_ids) / len(nl)
    mean_sql_len = sum(len(ids) for ids in sql_ids) / len(sql)
    nl_vocab = len(set(t for ids in nl_ids for t in ids))
    sql_vocab = len(set(t for ids in sql_ids for t in ids))

    print(f"{split}: mean_nl_len={mean_nl_len:.2f}, mean_sql_len={mean_sql_len:.2f}, "
          f"nl_vocab={nl_vocab}, sql_vocab={sql_vocab}")

# Check: after pre-processing, can every SQL query be recovered exactly?
train_sql = load_lines("data/train.sql")
bad = 0
for q in train_sql:
    ids = tokenizer(preprocess_sql(q)).input_ids
    decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if decoded.replace("lessthan", "<") != q:
        bad += 1
print(f"After pre-processing: {bad}/{len(train_sql)} SQL queries NOT recovered exactly")