_HIGHLIGHTS = "highlights"
_ARTICLE = "article"
_DEFAULT_VERSION = datasets.Version("3.0.0", "Using cased version.")
DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
END_TOKENS = [".", "!", "?", "...", "'", "`", '"', DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ")"]
# Set the maximum sequence lengths

# 
def _read_text_file(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines
  
# Put periods on the ends of lines that are missing 
def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."
      
# get highlight and article from a given data path
def _get_art_abs(data_path):
    lines = _read_text_file(data_path)
    lines = [fix_missing_period(line) for line in lines]
    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)
    # Make article into a single string
    article = " ".join(article_lines)
    abstract = " ".join(highlights)
    return article, abstract

if __name__ == "__main__":
    data_file_path = './sample_data'
    idx = 0
    articles = []
    highlights = []
    for file_name in os.scandir(data_file_path):
        if file_name.is_file():
            article, highlight = _get_art_abs(file_name.path)
            if not article or not highlight:
                continue
            articles.append(article)
            highlights.append(highlight)

    train_model(articles=articles, highlights=highlights)
