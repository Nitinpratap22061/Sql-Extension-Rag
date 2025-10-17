const askBtn = document.getElementById("askBtn");
const queryInput = document.getElementById("query");
const responseDiv = document.getElementById("response");

// --- Improved Markdown parser (supports bold, italic, inline code, tables, and line breaks) ---
function parseMarkdown(text) {
  // Handle tables first
  text = text.replace(
    /\|(.+\|)\s*\n\|?(-+\|)+\s*\n((\|.*\|\s*\n?)*)/g,
    (match, headerRow, _, bodyRows) => {
      const headers = headerRow
        .split("|")
        .filter(Boolean)
        .map(h => `<th>${h.trim()}</th>`)
        .join("");
      const rows = bodyRows
        .trim()
        .split("\n")
        .filter(Boolean)
        .map(r => {
          const cells = r
            .split("|")
            .filter(Boolean)
            .map(c => `<td>${c.trim()}</td>`)
            .join("");
          return `<tr>${cells}</tr>`;
        })
        .join("");
      return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    }
  );

  // Handle bold, italic, inline code, and line breaks
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // bold
    .replace(/\*(.*?)\*/g, "<em>$1</em>")             // italic
    .replace(/`(.*?)`/g, "<code>$1</code>")           // inline code
    .replace(/\n/g, "<br>");                          // line breaks
}

// --- Main click event ---
askBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();
  if (!query) {
    responseDiv.innerHTML = "<p class='error'>⚠️ Please enter a question.</p>";
    return;
  }

  responseDiv.innerHTML = `
    <div class="loader"></div>
    <p>Thinking...</p>
  `;

  try {
    const response = await fetch("http://127.0.0.1:8502/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) throw new Error("Server Error");

    const data = await response.json();
    const answer = data.answer || "No answer found.";
    responseDiv.innerHTML = `<div class="markdown-body">${parseMarkdown(answer)}</div>`;
  } catch (err) {
    responseDiv.innerHTML = "<p class='error'>❌ Unable to connect to backend. Please ensure it's running on port 8502.</p>";
  }
});

// --- Press Enter to send query ---
queryInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") askBtn.click();
});
