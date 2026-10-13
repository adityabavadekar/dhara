// Rough Devanagari → Latin transliteration so a typed "gosaikunda" finds "गोसाइँकुण्ड".
// Mirrors dhara.assistant.entities.latin; used only for matching, never shown.
const CONS: Record<string, string> = {
  क: 'k', ख: 'kh', ग: 'g', घ: 'gh', ङ: 'ng', च: 'ch', छ: 'chh', ज: 'j', झ: 'jh', ञ: 'ny', ट: 't', ठ: 'th', ड: 'd', ढ: 'dh',
  ण: 'n', त: 't', थ: 'th', द: 'd', ध: 'dh', न: 'n', प: 'p', फ: 'ph', ब: 'b', भ: 'bh', म: 'm', य: 'y', र: 'r', ल: 'l', व: 'w',
  श: 'sh', ष: 'sh', स: 's', ह: 'h',
}
const VOW: Record<string, string> = { अ: 'a', आ: 'a', इ: 'i', ई: 'i', उ: 'u', ऊ: 'u', ए: 'e', ऐ: 'ai', ओ: 'o', औ: 'au', ऋ: 'ri' }
const SIGN: Record<string, string> = { 'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ृ': 'ri' }
const NASAL: Record<string, string> = { 'ँ': 'n', 'ं': 'n', 'ः': 'h' }

export function latin(text: string): string {
  const c = [...text]
  let out = ''
  c.forEach((ch, i) => {
    const nxt = c[i + 1] ?? ''
    if (CONS[ch]) out += CONS[ch] + (CONS[nxt] || NASAL[nxt] ? 'a' : '')
    else if (VOW[ch]) out += VOW[ch]
    else if (SIGN[ch]) out += SIGN[ch]
    else if (NASAL[ch]) out += NASAL[ch]
    else if (ch !== '्') out += ch.toLowerCase()
  })
  return out
}

/** True when `name` contains `query`, directly or through its Latin spelling (vowels loosened). */
export function nameMatches(name: string, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  const n = name.toLowerCase()
  if (n.includes(q)) return true
  const loose = (s: string) => s.replace(/[aeiou]+/g, '').replace(/(.)\1+/g, '$1')
  const l = latin(name)
  return l.includes(q) || (q.length >= 3 && loose(l).includes(loose(q)))
}
