

def task_definer(task_type):
    if task_type == "quadruples":
        return (
            "Now you are an expert in extracting quadruples from a text. "
            "Given the input text, first determine if it contains any opinion expression. "
            "If no opinion exists, output 'statement-non-opinion'. "
            "If opinions exist, extract the quadruples. Note that: "
            "1) Target, Aspect, and Opinion MUST be explicitly found in the input text. "
            "2) Sentiment (pos, neg, or other) is determined based on the target, aspect, and opinion. "
            "3) Always return a JSON object: {\"quadruples\": [{\"target\": string, \"aspect\": string, \"opinion\": string, \"sentiment\": string}, ...]}\n"
        )
    