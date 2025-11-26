from docling.document_converter import DocumentConverter

source = "PhonePe_Transaction_Statement_unlocked.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)

markdown_text = result.document.export_to_markdown()

print(markdown_text)


def save_markdown(content, filename="output.md"):
    """
    Save text content to a Markdown (.md) file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


save_markdown(markdown_text, "phonepe-report.md")
