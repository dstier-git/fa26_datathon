"""Per-run Markdown report state."""

def table(df):
    return "```\n" + df.round(3).to_string() + "\n```"


class Report:
    def __init__(self):
        self.lines = []

    def log(self, message=""):
        print(message, flush=True)
        self.lines.append(message)

    def write(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
