import html
from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import HtmlFormatter

formatter = HtmlFormatter(nowrap=True, style="monokai")

def build_html_table(rows):
    # We fix the table layout so the columns stay the same width,
    # and we allow columns to wrap or scroll if needed.
    html_str = """
    <table style="width:100%; border-collapse: collapse; table-layout: fixed;" class="highlight">
      <thead>
        <tr>
          <th style="border: 1px solid black; padding: 6px; width:25%;">Prompt (C code)</th>
          <th style="border: 1px solid black; padding: 6px; width:25%;">Response</th>
          <th style="border: 1px solid black; padding: 6px; width:25%;">Extracted Answer</th>
          <th style="border: 1px solid black; padding: 6px; width:25%;">Correct Answer</th>
        </tr>
      </thead>
      <tbody>
    """

    for prompt, response, extracted, correct in rows:
        # --- Prompt (C code) with syntax highlighting ---
        prompt_html = highlight(prompt, CLexer(), formatter)
        # Wrap it in <pre> to preserve indentation, and in a <div> for scroll
        prompt_html = (
            "<div style='max-height:300px; overflow-y:auto;'>"
            f"<pre style='margin:0;'>{prompt_html}</pre>"
            "</div>"
        )

        # --- Response (plain text, but preserve newlines) ---
        response_escaped = html.escape(response)
        response_html = (
            "<div style='max-height:300px; overflow-y:auto;'>"
            f"<pre style='margin:0;'>{response_escaped}</pre>"
            "</div>"
        )

        # --- Extracted Answer ---
        extracted_escaped = html.escape(str(extracted))
        extracted_html = (
            "<div style='max-height:300px; overflow-y:auto;'>"
            f"<pre style='margin:0;'>{extracted_escaped}</pre>"
            "</div>"
        )

        # --- Correct Answer ---
        correct_escaped = html.escape(str(correct))
        correct_html = (
            "<div style='max-height:300px; overflow-y:auto;'>"
            f"<pre style='margin:0;'>{correct_escaped}</pre>"
            "</div>"
        )

        # Build the row
        html_str += f"""
        <tr>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{prompt_html}</td>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{response_html}</td>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{extracted_html}</td>
          <td style="border: 1px solid black; padding: 6px; vertical-align: top;">{correct_html}</td>
        </tr>
        """

    html_str += """
      </tbody>
    </table>
    """

    # Optionally embed the Pygments CSS for colored syntax:
    css = formatter.get_style_defs('.highlight')
    style_block = f"<style>{css}</style>"

    return style_block + html_str