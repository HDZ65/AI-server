export type MdChunk = { text: string; section?: string };

export function splitMarkdown(md: string, maxLen = 1200, overlap = 150): MdChunk[] {
  const lines = md.split(/\r?\n/);
  const chunks: MdChunk[] = [];
  let buf: string[] = [];
  let currentSection: string | undefined;

  const push = () => {
    if (!buf.length) return;
    const text = buf.join('\n').trim();
    if (text) chunks.push({ text, section: currentSection });
    buf = [];
  };

  for (const line of lines) {
    const h = line.match(/^#{2,6}\s+(.+)$/); // H2+
    if (h) {
      // démarre une nouvelle section
      push();
      currentSection = h[1].trim();
    }
    buf.push(line);
    if (buf.join('\n').length > maxLen + overlap) {
      // coupe en respectant une marge d'overlap
      const text = buf.join('\n');
      const parts: string[] = [];
      for (let i = 0; i < text.length; i += maxLen) {
        parts.push(text.slice(i, i + maxLen));
      }
      for (const p of parts) chunks.push({ text: p.trim(), section: currentSection });
      buf = [];
    }
  }
  push();
  return chunks;
}
