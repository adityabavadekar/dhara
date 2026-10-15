import { pinyin } from 'pinyin-pro'

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

// Phonetic skeleton for matching romanisations: ph→f, v→w, aspiration h dropped, nasal n before a
// consonant dropped (गोसाइँकुण्ड: "gosainkunda" = "gosaikunda"), vowels dropped, doubles merged.
const loose = (s: string) =>
  s
    .replace(/ph/g, 'f')
    .replace(/v/g, 'w')
    .replace(/([kgcjtdpbs])h/g, '$1')
    .replace(/n(?=[bcdfgjkpqstwxz])/g, '')
    .replace(/[aeiouy]+/g, '')
    .replace(/(.)\1+/g, '$1')
const HAN = /\p{Script=Han}/u

/** Search keys of one name, computed once per analysis: lower case, Latin spelling of Devanagari,
 *  pinyin of Chinese names (border villages), and the phonetic skeleton of each word. */
export type NameKeys = { exact: string; latin: string; words: string[] }
export function nameKeys(name: string): NameKeys {
  const exact = name.toLowerCase()
  let lat = latin(name)
  if (HAN.test(name)) lat = pinyin(name, { toneType: 'none', type: 'array' }).join('').toLowerCase()
  const ascii = lat.normalize('NFD').replace(/[^a-z0-9 ]/g, '')
  return { exact, latin: ascii, words: ascii.split(/\s+/).filter(Boolean).map(loose) }
}

const oneOff = (a: string, b: string) => {
  let d = 0
  for (let i = 0; i < b.length && d <= 1; i++) if (a[i] !== b[i]) d++
  return d <= 1
}

/** How well a query (Latin, Devanagari or Chinese) matches a name (+0.5 at a word start): 3 exact text, 2 Latin
 *  spelling, 1 phonetic word start (4+ letters), 0.5 one letter off (5+ letters), 0 no match. */
export function matchScore(keys: NameKeys, query: string): number {
  const q = query.trim().toLowerCase()
  if (!q) return 1
  const ql = nameKeys(q)
  const startsWord = (text: string, part: string) => text.split(/\s+/).some((w) => w.startsWith(part))
  if (keys.exact.includes(q)) return startsWord(keys.exact, q) ? 3.5 : 3
  if (ql.latin && keys.latin.includes(ql.latin)) return startsWord(keys.latin, ql.latin) ? 2.5 : 2
  const sk = ql.words.join('')
  if (q.length >= 4 && sk.length >= 2 && keys.words.some((w) => w.startsWith(sk))) return 1
  if (q.length >= 5 && sk.length >= 4 && keys.words.some((w) => w.length >= sk.length && oneOff(w, sk))) return 0.5
  return 0
}

/** Items ranked by match; one-letter-off matches only when nothing closer exists. */
export function searchRank<T>(index: readonly (readonly [T, NameKeys])[], query: string, limit: number): T[] {
  const scored = index.map(([x, k]) => [x, matchScore(k, query)] as const).filter(([, s]) => s > 0)
  const best = Math.max(0, ...scored.map(([, s]) => s))
  return scored
    .filter(([, s]) => s >= Math.min(best, 1))
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([x]) => x)
}

export function keysMatch(keys: NameKeys, query: string): boolean {
  return matchScore(keys, query) > 0
}
